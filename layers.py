from lasagne.layers import Layer


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
