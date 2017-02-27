from operator import mul
import itertools
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import SaveWeights
from utils import EarlyStopping, WeightsLogger
from lasagne import objectives
from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer, ConcatLayer, FlattenLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Pool3DDNNLayer, batch_norm_dnn
from layers import Unpooling3D, Transformer3DLayer, WeightedSumLayer
from lasagne import updates
from lasagne import nonlinearities
from lasagne.init import Constant
import objective_functions as objective_f
from iterators import Affine3DTransformBatchIterator, Affine3DTransformExpandBatchIterator
import numpy as np


def get_epoch_finished(name, patience):
    return [
        SaveWeights(name + 'model_weights.pkl', only_best=True, pickle=False),
        WeightsLogger(name + 'weights_log.pkl'),
        EarlyStopping(patience=patience)
    ]


def get_back_pathway(forward_pathway, multi_channel=True):
    # We create the backwards path of the encoder from the forward path
    # We need to mirror the configuration of the layers and change the pooling operators with unpooling,
    # and the convolutions with deconvolutions (convolutions with diferent padding). This definitions
    # match the values of the possible_layers dictionary
    back_pathway = ''.join(['d' if l is 'c' else 'u' for l in forward_pathway[::-1]])
    last_conv = back_pathway.rfind('d')
    final_conv = 'f' if multi_channel else 'fU'
    back_pathway = back_pathway[:last_conv] + final_conv + back_pathway[last_conv + 1:]

    return back_pathway


def get_layers_string(
        net_layers,
        input_shape,
        convo_size=3,
        pool_size=2,
        dense_size=256,
        number_filters=32,
        multi_channel=True,
        padding='valid'
):
    input_shape_single = tuple(input_shape[:1] + (1,) + input_shape[2:])
    channels = range(0, input_shape[1])
    previous_layer = InputLayer(name='\033[30minput\033[0m', shape=input_shape) if multi_channel\
        else [InputLayer(name='\033[30minput_%d\033[0m' % i, shape=input_shape_single) for i in channels]

    convolutions = dict()
    c_index = 1
    p_index = 1
    c_size = (convo_size, convo_size, convo_size)
    for layer in net_layers:
        if layer == 'c':
            conv_layer = batch_norm_dnn(
                layer=Conv3DDNNLayer(
                    incoming=previous_layer,
                    name='\033[34mconv%d\033[0m' % c_index,
                    num_filters=number_filters,
                    filter_size=c_size,
                    pad=padding
                ),
                name='norm%d' % c_index
            ) if multi_channel else [batch_norm_dnn(
                layer=Conv3DDNNLayer(
                    incoming=layer,
                    name='\033[34mconv%d_%d\033[0m' % (c_index, i),
                    num_filters=number_filters,
                    filter_size=c_size,
                    pad=padding
                ),
                name='norm%d_%d' % (c_index, i)
            ) for layer, i in zip(previous_layer, channels)]
            convolutions['conv%d' % c_index] = conv_layer
            previous_layer = conv_layer
            c_index += 1
        elif layer == 't':
            b = np.zeros((3, 4), dtype='float32')
            b[0, 0] = 1
            b[1, 1] = 1
            b[2, 2] = 1
            w = Constant(0.0)
            previous_layer = Transformer3DLayer(
                localization_network=DenseLayer(
                    incoming=previous_layer,
                    name='\033[33mloc_net\033[0m',
                    num_units=12,
                    W=w,
                    b=b.flatten,
                    nonlinearity=None
                ),
                incoming=previous_layer,
                name='\033[33mtransf\033[0m',
            ) if multi_channel else [Transformer3DLayer(
                localization_network=DenseLayer(
                    incoming=previous_layer,
                    name='\033[33mloc_net_%d\033[0m' % i,
                    num_units=12,
                    W=w,
                    b=b.flatten,
                    nonlinearity=None
                ),
                incoming=layer,
                name='\033[33mtransf_%d\033[0m' % i,
            ) for layer, i in zip(previous_layer, channels)]
        elif layer == 'a':
            previous_layer = Pool3DDNNLayer(
                incoming=previous_layer,
                name='\033[31mavg_pool%d\033[0m' % p_index,
                pool_size=pool_size,
                mode='average_inc_pad'
            ) if multi_channel else [Pool3DDNNLayer(
                incoming=layer,
                name='\033[31mavg_pool%d_%d\033[0m' % (p_index, i),
                pool_size=pool_size,
                mode='average_inc_pad'
            ) for layer, i in zip(previous_layer, channels)]
            p_index += 1
        elif layer == 'm':
            previous_layer = MaxPool3DDNNLayer(
                incoming=previous_layer,
                name='\033[31mmax_pool%d\033[0m' % p_index,
                pool_size=pool_size
            ) if multi_channel else [MaxPool3DDNNLayer(
                incoming=layer,
                name='\033[31mmax_pool%d_%d\033[0m' % (p_index, i),
                pool_size=pool_size
            ) for layer, i in zip(previous_layer, channels)]
            p_index += 1
        elif layer == 'u':
            p_index -= 1
            previous_layer = Unpooling3D(
                incoming=previous_layer,
                name='\033[35munpool%d\033[0m' % p_index,
                pool_size=pool_size
            ) if multi_channel else [Unpooling3D(
                incoming=layer,
                name='\033[35munpool%d_%d\033[0m' % (p_index, i),
                pool_size=pool_size
            ) for layer, i in zip(previous_layer, channels)]
        elif layer == 's':
            previous_layer = ElemwiseSumLayer(
                incomings=[convolutions['conv%d' % (c_index - 1)], previous_layer],
                name='short%d' % (c_index - 1)
            ) if multi_channel else [ElemwiseSumLayer(
                incomings=[convolutional, layer],
                name='short%d_%d' % (c_index - 1, i)
            ) for convolutional, layer, i in zip(convolutions['conv%d' % (c_index - 1)], previous_layer, channels)]
        elif layer == 'd':
            c_index -= 1
            previous_layer = batch_norm_dnn(
                layer=Conv3DDNNLayer(
                    incoming=previous_layer,
                    name='\033[36mdeconv%d\033[0m' % c_index,
                    num_filters=number_filters,
                    filter_size=c_size,
                    W=convolutions['conv%d' % (c_index - 1)].W.T,
                    pad='full'
                ),
                name='denorm%d' % c_index
            ) if multi_channel else [batch_norm_dnn(
                layer=Conv3DDNNLayer(
                    incoming=layer,
                    name='\033[36mdeconv%d_%d\033[0m' % (c_index, i),
                    num_filters=number_filters,
                    filter_size=c_size,
                    W=convolutional['conv%d' % (c_index - 1)].W.T,
                    pad='full'
                ),
                name='denorm%d_%d' % (c_index, i)
            ) for convolutional, layer, i in zip(convolutions['conv%d' % (c_index - 1)], previous_layer, channels)]
        elif layer == 'o':
            previous_layer = DropoutLayer(
                incoming=previous_layer,
                name='drop%d' % (c_index - 1),
                p=0.5
            ) if multi_channel else [DropoutLayer(
                incoming=layer,
                name='drop%d_%d' % (c_index - 1, i),
                p=0.5
            ) for layer, i in zip(previous_layer, channels)]
        elif layer == 'f':
            c_index -= 1
            previous_layer = Conv3DDNNLayer(
                incoming=previous_layer,
                name='\033[36mfinal\033[0m',
                num_filters=input_shape[1],
                filter_size=c_size,
                pad='full'
            ) if multi_channel else [Conv3DDNNLayer(
                incoming=layer,
                name='\033[36mfinal_%d\033[0m' % i,
                num_filters=1,
                filter_size=c_size,
                pad='full'
            ) for layer, i in zip(previous_layer, channels)]
        elif layer == 'r':
            previous_layer = ReshapeLayer(
                incoming=previous_layer,
                name='\033[32mreshape\033[0m',
                shape=([0], -1)
            )
        elif layer == 'U':
            # Multichannel-only layer
            previous_layer = ConcatLayer(
                incomings=previous_layer,
                name='\033[32munion\033[0m'
            )
        elif layer == 'D':
            previous_layer = DenseLayer(
                incoming=previous_layer,
                name='\033[32mdense\033[0m',
                num_units=dense_size,
                nonlinearity=nonlinearities.softmax
            )
        elif layer == 'S':
            previous_layer = DenseLayer(
                incoming=previous_layer,
                name='\033[32m3d_out\033[0m',
                num_units=reduce(mul, input_shape[2:], 1),
                nonlinearity=nonlinearities.softmax
            )
        elif layer == 'C':
            previous_layer = DenseLayer(
                incoming=previous_layer,
                name='\033[32mclass_out\033[0m',
                num_units=2,
                nonlinearity=nonlinearities.softmax
            )

    return previous_layer


def get_layers_registration(
        input_shape,
        convo_blocks=2,
        convo_size=3,
        pool_size=2,
        number_filters=32
):
    source_input = InputLayer(name='\033[30mbaseline\033[0m', shape=(None, 1) + tuple(input_shape))
    source = source_input
    target = InputLayer(name='\033[30mfollow\033[0m', shape=(None, 1) + tuple(input_shape))

    for i in range(convo_blocks):
        source, target = get_shared_convolutional_block(
            source,
            target,
            convo_size=convo_size,
            num_filters=number_filters,
            pool_size=pool_size,
        )

    b = np.zeros((3, 4), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    b[2, 2] = 1
    w = Constant(0.0)
    register = Transformer3DLayer(
        localization_network=DenseLayer(
            incoming=ConcatLayer(
                incomings=[source, target],
                name='union'
            ),
            name='\033[33mloc_net\033[0m',
            num_units=12,
            W=w,
            b=b.flatten,
            nonlinearity=None
        ),
        incoming=source_input,
        name='\033[33mtransf\033[0m'
    )
    output = FlattenLayer(
        incoming=register,
        name='\033[32m3d_out\033[0m',
    )
    return output


def get_layers_greenspan(
        input_channels,
):
    input_shape = (None, input_channels, 32, 32)
    images = ['axial', 'coronal', 'sagital']
    baseline = [InputLayer(name='\033[30mbaseline_%s\033[0m' % i, shape=input_shape) for i in images]
    followup = [InputLayer(name='\033[30mfollow_%s\033[0m' % i, shape=input_shape) for i in images]
    lnets = [get_lnet(b, f, i) for b, f, i in zip(baseline, followup, images)]
    union = ConcatLayer(
        incomings=lnets,
        name='concat_final'
    )
    dense = DenseLayer(
        incoming=union,
        name='\033[32mdense_final\033[0m',
        num_units=16,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    softmax = DenseLayer(
        incoming=dense,
        name='\033[32mclass_out\033[0m',
        num_units=2,
        nonlinearity=nonlinearities.softmax
    )
    return softmax


def get_convolutional_longitudinal(
    convo_blocks,
    input_shape,
    images,
    convo_size,
    pool_size,
    number_filters,
    padding,
    drop,
    register
):
    if not isinstance(convo_size, list):
        convo_size = [convo_size] * convo_blocks

    if not isinstance(number_filters, list):
        number_filters = [number_filters] * convo_blocks
    input_shape_single = tuple(input_shape[:1] + (1,) + input_shape[2:])
    channels = input_shape[1]
    if not images:
        images = ['im%d' % i for i in range(channels / 2)]
    baseline = [InputLayer(name='\033[30mbaseline_%s\033[0m' % i, shape=input_shape_single) for i in images]
    followup = [InputLayer(name='\033[30mfollow_%s\033[0m' % i, shape=input_shape_single) for i in images]

    if register:
        b = np.zeros((3, 4), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        b[2, 2] = 1
        w = Constant(0.0)
        followup = [Transformer3DLayer(
            localization_network=DenseLayer(
                incoming=ConcatLayer(
                    incomings=[p1, p2]
                ),
                name='\033[33mloc_net\033[0m',
                num_units=12,
                W=w,
                b=b.flatten,
                nonlinearity=None
            ),
            incoming=p1,
            name='\033[33mtransf\033[0m',
        ) for p1, p2, i in zip(baseline, followup, images)]

    sub_counter = itertools.count()
    convo_counter = itertools.count()
    subconvo_counter = itertools.count()

    subtraction = [WeightedSumLayer(
        name='subtraction_init_%s' % i,
        incomings=[p1, p2]
    ) for p1, p2, i in zip(baseline, followup, images)]

    for c, f in zip(convo_size, number_filters):
        baseline, followup = zip(*[get_shared_convolutional_block(
            p1,
            p2,
            c,
            f,
            pool_size,
            drop,
            padding,
            sufix=i,
            counter=convo_counter
        ) for p1, p2, i in zip(baseline, followup, images)])
        index = sub_counter.next()
        subtraction = [ElemwiseSumLayer(
            name='subtraction_%s%d' % (i, index),
            incomings=[
                get_convolutional_block(
                    s,
                    c,
                    f,
                    pool_size,
                    drop,
                    padding,
                    sufix=i,
                    counter=subconvo_counter
                ),
                WeightedSumLayer(
                    name='wsubtraction_%s%d' % (i, index),
                    incomings=[p1, p2]
                )
            ]
        ) for p1, p2, s, i, in zip(baseline, followup, subtraction, images)]

    return baseline, followup, subtraction


def get_layers_longitudinal(
        convo_blocks,
        input_shape,
        images=None,
        convo_size=3,
        pool_size=2,
        dense_size=256,
        number_filters=32,
        padding='valid',
        drop=0.5,
        register=False,
):
    baseline, followup, subtraction = get_convolutional_longitudinal(
        convo_blocks,
        input_shape,
        images,
        convo_size,
        pool_size,
        number_filters,
        padding,
        drop,
        register
    )

    image_union = [ConcatLayer(
        incomings=[FlattenLayer(b), FlattenLayer(s)],
        name='union'
    ) for b, f, s in zip(baseline, subtraction)]

    dense = [DenseLayer(
        incoming=u,
        name='\033[32mdense_%s\033[0m' % i,
        num_units=dense_size,
        nonlinearity=nonlinearities.softmax
    ) for u, i in zip(image_union, images)]

    union = ConcatLayer(
        incomings=dense,
        name='union'
    )

    soft = DenseLayer(
        incoming=union,
        name='\033[32mclass_out\033[0m',
        num_units=2,
        nonlinearity=nonlinearities.softmax
    )

    return soft


def get_layers_longitudinal_deformation(
            convo_blocks,
            input_shape,
            images=None,
            convo_size=3,
            pool_size=2,
            dense_size=256,
            number_filters=32,
            padding='valid',
            drop=0.5,
            register=False,
):
    if not isinstance(convo_size, list):
        convo_size = [convo_size] * convo_blocks

    if not isinstance(number_filters, list):
        number_filters = [number_filters] * convo_blocks

    baseline, followup, subtraction = get_convolutional_longitudinal(
        convo_blocks,
        input_shape,
        images,
        convo_size,
        pool_size,
        number_filters,
        padding,
        drop,
        register
    )

    defo_input_shape = (input_shape[:1] + (3,) + (convo_blocks*2+1, convo_blocks*2+1, convo_blocks*2+1))
    deformation = [InputLayer(name='\033[30mdeformation_%s\033[0m' % i, shape=defo_input_shape) for i in images]

    defo_counter = itertools.count()
    for c, f in zip(convo_size, number_filters):
        deformation = [get_convolutional_block(
            d,
            c,
            f,
            pool_size,
            drop,
            padding,
            sufix=i,
            counter=defo_counter
        ) for d, i in zip(deformation, images)]

    image_union = [ConcatLayer(
        incomings=[FlattenLayer(b), FlattenLayer(s), FlattenLayer(d)],
        name='union'
    ) for b, s, d in zip(baseline, subtraction, deformation)]

    dense = [DenseLayer(
        incoming=u,
        name='\033[32mdense_%s\033[0m' % i,
        num_units=dense_size,
        nonlinearity=nonlinearities.softmax
    ) for u, i in zip(image_union, images)]

    union = ConcatLayer(
        incomings=dense,
        name='union'
    )

    soft = DenseLayer(
        incoming=union,
        name='\033[32mclass_out\033[0m',
        num_units=2,
        nonlinearity=nonlinearities.softmax
    )

    return soft


def get_convolutional_block(
        incoming,
        convo_size=3,
        num_filters=32,
        pool_size=2,
        drop=0.5,
        padding='valid',
        counter=itertools.count(),
        sufix=''
):
    index = counter.next()
    convolution = Conv3DDNNLayer(
        incoming=incoming,
        name='\033[34mconv_%s%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        pad=padding
    )
    normalisation = batch_norm_dnn(
        layer=convolution,
        name='norm_%s%d' % (sufix, index)
    )
    dropout = DropoutLayer(
        incoming=normalisation,
        name='drop_%s%d' % (sufix, index),
        p=drop
    )
    pool = Pool3DDNNLayer(
        incoming=dropout,
        name='\033[31mavg_pool_%s%d\033[0m' % (sufix, index),
        pool_size=pool_size,
        mode='average_inc_pad'
    )

    return pool


def get_shared_convolutional_block(
            incoming1,
            incoming2,
            convo_size=3,
            num_filters=32,
            pool_size=2,
            drop=0.5,
            padding='valid',
            counter=itertools.count(),
            sufix=''
):

    index = counter.next()

    convolution1 = Conv3DDNNLayer(
        incoming=incoming1,
        name='\033[34mconv_%s1_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        pad=padding
    )
    convolution2 = Conv3DDNNLayer(
        incoming=incoming2,
        name='\033[34mconv_%s2_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        W=convolution1.W,
        b=convolution1.b,
        pad=padding
    )
    normalisation1 = batch_norm_dnn(
        layer=convolution1,
        name='norm_%s1_%d' % (sufix, index)
    )
    normalisation2 = batch_norm_dnn(
        layer=convolution2,
        name='norm_%s2_%d' % (sufix, index)
    )
    dropout1 = DropoutLayer(
        incoming=normalisation1,
        name='drop_%s1_%d' % (sufix, index),
        p=drop
    )
    dropout2 = DropoutLayer(
        incoming=normalisation2,
        name='drop_%s2_%d' % (sufix, index),
        p=drop
    )
    pool1 = Pool3DDNNLayer(
        incoming=dropout1,
        name='\033[31mavg_pool_%s1_%d\033[0m' % (sufix, index),
        pool_size=pool_size,
        mode='average_inc_pad'
    )
    pool2 = Pool3DDNNLayer(
        incoming=dropout2,
        name='\033[31mavg_pool_%s2_%d\033[0m' % (sufix, index),
        pool_size=pool_size,
        mode='average_inc_pad'
    )

    return pool1, pool2


def get_convolutional_block2d(
            incoming,
            convo_size=3,
            num_filters=32,
            pool_size=2,
            drop=0.5,
            padding='valid',
            counter=itertools.count(),
            sufix=''
):
        index = counter.next()

        convolution = Conv2DLayer(
            incoming=incoming,
            name='\033[34mconv_%s%d\033[0m' % (sufix, index),
            num_filters=num_filters,
            filter_size=convo_size,
            pad=padding
        )
        normalisation = batch_norm_dnn(
            layer=convolution,
            name='norm_%s%d' % (sufix, index)
        )
        dropout = DropoutLayer(
            incoming=normalisation,
            name='drop_%s%d' % (sufix, index),
            p=drop
        )
        pool = MaxPool2DLayer(
            incoming=dropout,
            name='\033[31mavg_pool_%s%d\033[0m' % (sufix, index),
            pool_size=pool_size,
            mode='average_inc_pad'
        )

        return pool


def get_shared_convolutional_block2d(
        incoming1,
        incoming2,
        convo_size=3,
        num_filters=32,
        pool_size=2,
        drop=0.5,
        padding='valid',
        counter=itertools.count(),
        sufix='',
        nonlinearity=nonlinearities.very_leaky_rectify
):
    index = counter.next()

    convolution1 = Conv2DLayer(
        incoming=incoming1,
        name='\033[34mconv_%s1_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        pad=padding,
        nonlinearity=nonlinearity
    )
    convolution2 = Conv2DLayer(
        incoming=incoming2,
        name='\033[34mconv_%s2_%d\033[0m' % (sufix, index),
        num_filters=num_filters,
        filter_size=convo_size,
        W=convolution1.W,
        b=convolution1.b,
        pad=padding,
        nonlinearity=nonlinearity
    )
    dropout1 = DropoutLayer(
        incoming=convolution1,
        name='drop_%s1_%d' % (sufix, index),
        p=drop
    )
    dropout2 = DropoutLayer(
        incoming=convolution2,
        name='drop_%s2_%d' % (sufix, index),
        p=drop
    )
    pool1 = MaxPool2DLayer(
        incoming=dropout1,
        name='\033[31mmax_pool_%s1_%d\033[0m' % (sufix, index),
        pool_size=pool_size
    )
    pool2 = MaxPool2DLayer(
        incoming=dropout2,
        name='\033[31mmax_pool_%s2_%d\033[0m' % (sufix, index),
        pool_size=pool_size
    )

    return pool1, pool2


def get_lnet(in1, in2, sufix):
    counter = itertools.count()
    vnet1_1, vnet1_2 = get_shared_convolutional_block2d(in1, in2, 5, 24, sufix=sufix, drop=0.25, counter=counter)
    vnet2_1, vnet2_2 = get_shared_convolutional_block2d(vnet1_1, vnet1_2, 3, 32, sufix=sufix, drop=0.25)
    index = counter.next()
    convolution1 = Conv2DLayer(
        incoming=vnet2_1,
        name='\033[34mconv_%s1_%d\033[0m' % (sufix, index),
        num_filters=48,
        filter_size=3,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    convolution2 = Conv2DLayer(
        incoming=vnet2_2,
        name='\033[34mconv_%s2_%d\033[0m' % (sufix, index),
        num_filters=48,
        filter_size=3,
        W=convolution1.W,
        b=convolution1.b,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    dropout1 = DropoutLayer(
        incoming=convolution1,
        name='drop_%s1_%d' % (sufix, index),
        p=0.25
    )
    dropout2 = DropoutLayer(
        incoming=convolution2,
        name='drop_%s2_%d' % (sufix, index),
        p=0.25
    )
    union = ConcatLayer(
        incomings=[dropout1, dropout2],
        name='concat_%s' % sufix
    )
    convolutionf = Conv2DLayer(
        incoming=union,
        name='\033[34mconv_%sf\033[0m' % sufix,
        num_filters=48,
        filter_size=1,
        nonlinearity=nonlinearities.very_leaky_rectify
    )
    dense = DenseLayer(
        incoming=convolutionf,
        name='\033[32mlnet_%s_out\033[0m' % sufix,
        num_units=16,
        nonlinearity=nonlinearities.very_leaky_rectify
    )

    return dense


def create_classifier_net(
        layers,
        patience,
        name,
        obj_f='xent',
        epochs=200
):

    objective_function = {
        'xent': objectives.categorical_crossentropy,
        'pdsc': objective_f.probabilistic_dsc_objective,
        'ldsc': objective_f.logarithmic_dsc_objective
    }

    return NeuralNet(

        layers=layers,

        regression=False,
        objective_loss_function=objective_function[obj_f],
        custom_scores=[
            ('prob dsc', objective_f.accuracy_dsc_probabilistic),
            ('dsc', objective_f.accuracy_dsc),
        ],

        # update=updates.adadelta,
        update=updates.adam,
        update_learning_rate=1e-4,

        on_epoch_finished=get_epoch_finished(name, patience),

        batch_iterator_train=BatchIterator(batch_size=512),

        verbose=11,
        max_epochs=epochs
    )


def create_segmentation_net(
        layers,
        patience,
        name,
        custom_scores=None,
        epochs=200
):
    return NeuralNet(

        layers=layers,

        regression=True,

        update=updates.adam,
        update_learning_rate=1e-3,

        on_epoch_finished=get_epoch_finished(name, patience),

        custom_scores=custom_scores,

        objective_loss_function=objectives.categorical_crossentropy,

        verbose=11,
        max_epochs=epochs
    )


def create_registration_net(
            layers,
            patience,
            name,
            batch_iterator=Affine3DTransformExpandBatchIterator(
                batch_size=64,
                input_layers=['\033[30mbaseline\033[0m']
            ),
            custom_scores=None,
            epochs=200
):
        return NeuralNet(

            layers=layers,

            regression=True,

            update=updates.adadelta,
            # update_learning_rate=1e-3,

            on_epoch_finished=get_epoch_finished(name, patience),

            custom_scores=custom_scores,

            objective_loss_function=objective_f.cross_correlation,

            batch_iterator_train=batch_iterator,

            verbose=11,
            max_epochs=epochs
        )


def create_cnn3d_det_string(
            cnn_path,
            input_shape,
            convo_size,
            padding,
            pool_size,
            dense_size,
            number_filters,
            patience,
            multichannel,
            name,
            epochs
):

    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'rC' if multichannel else 'rUC'
    final_layers = cnn_path.replace('a', 'ao').replace('m', 'mo') + final_layers

    layer_list = get_layers_string(
        net_layers=final_layers,
        input_shape=input_shape,
        convo_size=convo_size,
        pool_size=pool_size,
        dense_size=dense_size,
        number_filters=number_filters,
        multi_channel=multichannel,
        padding=padding
    )

    return create_classifier_net(
        layer_list,
        patience,
        name,
        epochs=epochs
    )


def create_cnn3d_longitudinal(
        convo_blocks,
        input_shape,
        images,
        convo_size,
        pool_size,
        dense_size,
        number_filters,
        padding,
        drop,
        register,
        defo,
        patience,
        name,
        epochs
):
    layer_list = get_layers_longitudinal(
        convo_blocks=convo_blocks,
        input_shape=input_shape,
        images=images,
        convo_size=convo_size,
        pool_size=pool_size,
        dense_size=dense_size,
        number_filters=number_filters,
        padding=padding,
        drop=drop,
        register=register
    ) if not defo else get_layers_longitudinal_deformation(
        convo_blocks=convo_blocks,
        input_shape=input_shape,
        images=images,
        convo_size=convo_size,
        pool_size=pool_size,
        dense_size=dense_size,
        number_filters=number_filters,
        padding=padding,
        drop=drop,
        register=register
    )

    return create_classifier_net(
        layer_list,
        patience,
        name,
        epochs=epochs
    )


def create_cnn_greenspan(
            input_channels,
            patience,
            name,
            epochs
):
        layer_list = get_layers_greenspan(input_channels)

        return create_classifier_net(
            layer_list,
            patience,
            name,
            epochs=epochs
        )


def create_cnn3d_register(
            input_shape,
            convo_size,
            convo_blocks,
            pool_size,
            number_filters,
            data_augment_p,
            patience,
            name,
            epochs
):
    layer_list = get_layers_registration(
        input_shape=input_shape,
        convo_size=convo_size,
        convo_blocks=convo_blocks,
        pool_size=pool_size,
        number_filters=number_filters
    )

    return create_registration_net(
        layer_list,
        patience,
        name,
        epochs=epochs,
        batch_iterator=Affine3DTransformBatchIterator(
                affine_p=data_augment_p,
                batch_size=32,
                input_layers=['\033[30mbaseline\033[0m']
            )
    )


def create_unet3d_det_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        multichannel,
        name,
        epochs
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_path + get_back_pathway(forward_path, multichannel) + 'r' + 'C'

    layer_list = get_layers_string(
        net_layers=final_layers,
        input_shape=input_shape,
        convo_size=convo_size,
        pool_size=pool_size,
        number_filters=number_filters,
        multi_channel=multichannel
    )

    return create_classifier_net(
        layer_list,
        patience,
        name,
        epochs=epochs
    )


def create_unet3d_seg_string(
            forward_path,
            input_shape,
            convo_size,
            pool_size,
            number_filters,
            patience,
            multichannel,
            name,
            epochs
):

    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_path + get_back_pathway(forward_path, multichannel) + 'r' + 'S'

    layer_list = get_layers_string(
        net_layers=final_layers,
        input_shape=input_shape,
        convo_size=convo_size,
        pool_size=pool_size,
        number_filters=number_filters,
        multi_channel=multichannel
    )

    return create_segmentation_net(
        layer_list,
        patience,
        name,
        epochs=epochs
    )


def create_unet3d_shortcuts_det_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        multichannel,
        name,
        epochs
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    back_pathway = get_back_pathway(forward_path, multichannel).replace('d', 'sd').replace('f', 'sf')
    final_layers = (forward_path + back_pathway + 'r' + 'C').replace('csd', 'cd')

    layer_list = get_layers_string(
        net_layers=final_layers,
        input_shape=input_shape,
        convo_size=convo_size,
        pool_size=pool_size,
        number_filters=number_filters,
        multi_channel=multichannel
    )

    return create_classifier_net(
        layer_list,
        patience,
        name,
        epochs=epochs
    )


def create_unet3d_shortcuts_seg_string(
            forward_path,
            input_shape,
            convo_size,
            pool_size,
            number_filters,
            patience,
            multichannel,
            name,
            epochs
):

    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    back_pathway = get_back_pathway(forward_path, multichannel).replace('d', 'sd').replace('f', 'sf')
    final_layers = (forward_path + back_pathway + 'r' + 'S').replace('csd', 'cd')

    layer_list = get_layers_string(
        net_layers=final_layers,
        input_shape=input_shape,
        convo_size=convo_size,
        pool_size=pool_size,
        number_filters=number_filters,
        multi_channel=multichannel
    )

    return create_segmentation_net(
        layer_list,
        patience,
        multichannel,
        name,
        epochs=epochs
    )


def create_encoder3d_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        multichannel,
        name,
        epochs=200
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = forward_path + get_back_pathway(forward_path, multichannel) + 'r'

    encoder = NeuralNet(
        layers=get_layers_string(final_layers, input_shape, convo_size, pool_size, number_filters, multichannel),

        regression=True,

        update=updates.adam,
        update_learning_rate=1e-3,

        on_epoch_finished=get_epoch_finished(name, patience),

        verbose=11,
        max_epochs=epochs
    )

    return encoder
