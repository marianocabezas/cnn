import theano
import theano.tensor as T

from lasagne.init import Constant
from lasagne.layers import Layer, MergeLayer
from lasagne.utils import as_tuple


class Unpooling3D(Layer):
    def __init__(self, pool_size=2, ignore_border=True, **kwargs):
        super(Unpooling3D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.ignore_border = ignore_border

    def get_output_for(self, data, **kwargs):
        output = data.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3).repeat(self.pool_size, axis=4)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + tuple(a * self.pool_size for a in input_shape[2:])


class WeightedSumLayer(MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(WeightedSumLayer, self).__init__(incomings, **kwargs)
        self.coeff_left = self.add_param(Constant(-1), (1,), name='coeff_left')
        self.coeff_right = self.add_param(Constant(1), (1,), name='coeff_right')

    def get_params(self, unwrap_shared=True, **tags):
        return [self.coeff_left, self.coeff_right]

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        left, right = inputs
        return left * self.coeff_left + right * self.coeff_right


class Transformer3DLayer(MergeLayer):
    """
    Spatial transformer layer
    The layer applies an affine transformation on the input. The affine
    transformation is parameterized with twelve learned parameters [1]_.
    The output is interpolated with a bilinear transformation.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns, input_depth)``.
    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the affine
        transformation. See the example for how to initialize to the identity
        transform.
    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.
    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    Examples
    --------
    Here we set up the layer to initially do the identity transform, similarly
    to [1]_. Note that you will want to use a localization with linear output.
    If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
    then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
    t6 move the center position.
    >>> import numpy as np
    >>> import lasagne
    >>> b = np.zeros((2, 3), dtype='float32')
    >>> b[0, 0] = 1
    >>> b[1, 1] = 1
    >>> b[2, 2] = 1
    >>> b = b.flatten()  # identity transform
    >>> W = lasagne.init.Constant(0.0)
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=12, W=W, b=b,
    ... nonlinearity=None)
    >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
    """

    def __init__(self, incoming, localization_network, downsample_factor=1,
                 **kwargs):
        super(Transformer3DLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.downsample_factor = as_tuple(downsample_factor, 3)

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 12 or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, 12)")
        if len(input_shp) != 5:
            raise ValueError("The input network must have a 5-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns, input_depth)")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input_l, theta = inputs
        return _transform_affine(theta, input_l, self.downsample_factor)


def _transform_affine(theta, input_l, downsample_factor):
    num_batch, num_channels, height, width, depth = input_l.shape
    theta = T.reshape(theta, (-1, 3, 4))

    # grid of (x_t, y_t, z_t, 1), eq (1) in ref [1]
    out_height = T.cast(height // downsample_factor[0], 'int64')
    out_width = T.cast(width // downsample_factor[1], 'int64')
    out_depth = T.cast(width // downsample_factor[2], 'int64')
    grid = _meshgrid(out_height, out_width, out_depth)

    # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    z_s = T_g[:, 2]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()
    z_s_flat = z_s.flatten()

    # dimshuffle input to  (bs, height, width, depth, channels)
    input_dim = input.dimshuffle(0, 2, 3, 4, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat, z_s_flat,
        out_height, out_width, out_depth)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, out_depth, num_channels))
    output = output.dimshuffle(0, 4, 1, 2, 3)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, z, out_height, out_width, out_depth):
    # *_f are floats
    num_batch, height, width, depth, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)
    depth_f = T.cast(depth, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)
    z = T.clip(z, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height/depth - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)
    z = (z + 1) / 2 * (depth_f - 1)

    # obtain indices of the 2x2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    z0_f = T.floor(z)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    z1_f = z0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    z0 = T.cast(z0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')
    z1 = T.cast(T.minimum(z1_f, depth_f - 1), 'int64')

    # The input is [num_batch, height, width, depth, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width*depth, channels]. We need
    # to offset all indices to match the flat version
    dim1 = height * width * depth
    dim2 = width * depth
    dim3 = depth
    base = T.repeat(
        T.arange(num_batch, dtype='int64') * dim1, out_height * out_width * out_depth)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    base_x0 = x0 * dim3
    base_x1 = x1 * dim3
    idx_a = base_y0 + base_x0 + z0
    idx_b = base_y1 + base_x0 + z0
    idx_c = base_y0 + base_x1 + z0
    idx_d = base_y1 + base_x1 + z0
    idx_e = base_y0 + base_x0 + z1
    idx_f = base_y1 + base_x0 + z1
    idx_g = base_y0 + base_x1 + z1
    idx_h = base_y1 + base_x1 + z1


    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]
    Ie = im_flat[idx_e]
    If = im_flat[idx_f]
    Ig = im_flat[idx_g]
    Ih = im_flat[idx_h]

    # calculate interpolated values
    wa = ((x1_f - x) * (y1_f - y) * (z1_f - z)).dimshuffle(0, 'x')
    wb = ((x1_f - x) * (y - y0_f) * (z1_f - z)).dimshuffle(0, 'x')
    wc = ((x - x0_f) * (y1_f - y) * (z1_f - z)).dimshuffle(0, 'x')
    wd = ((x - x0_f) * (y - y0_f) * (z1_f - z)).dimshuffle(0, 'x')
    we = ((x1_f - x) * (y1_f - y) * (z0_f - z)).dimshuffle(0, 'x')
    wf = ((x1_f - x) * (y - y0_f) * (z0_f - z)).dimshuffle(0, 'x')
    wg = ((x - x0_f) * (y1_f - y) * (z0_f - z)).dimshuffle(0, 'x')
    wh = ((x - x0_f) * (y - y0_f) * (z0_f - z)).dimshuffle(0, 'x')
    output = T.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id, we * Ie, wf * If, wg *Ig, wh * Ih], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop - start) / (num - 1)
    return T.arange(num, dtype=theano.config.floatX) * step + start


def _meshgrid(height, width, depth):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t,z_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(
        T.reshape(T.dot(
            _linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
            T.ones((1, width))), (height, width, 1)),
        T.ones((1, 1, depth))
    )
    y_t = T.dot(
        T.reshape(T.dot(
            T.ones((height, 1)),
            _linspace(-1.0, 1.0, width).dimshuffle('x', 0)), (height, width, 1)),
        T.ones((1, 1, depth))
    )
    z_t = T.dot(T.ones((height, width, 1)), T.reshape(_linspace(-1.0, 1.0, depth), (1, 1, -1)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    z_t_flat = z_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, z_t_flat, ones], axis=0)
    return grid
