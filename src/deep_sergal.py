import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy
from theano.gpuarray.tests.test_basic_ops import rng

class DeepSergal:
    def CNNTest(self, input_img):
        rng = numpy.random.RandomState(101) # init rng
        
        # instantiate 4D tensor for input
        inp = T.tensor4(name='input')
        
        # number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width
        W_shape = (2, 3, 9, 9)
        W_bound = numpy.sqrt(W_shape[1] * W_shape[2] * W_shape[3])
        
        W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / W_bound,
                high=1.0 / W_bound,
                size=W_shape),
            dtype=inp.dtype), name ='W')
        
        b_shp = (2,)
        b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=inp.dtype), name ='b')
        
        conv_out = conv2d(inp, W)
        
        output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
        
        f = theano.function([inp], output)
        
        return f(input_img)
