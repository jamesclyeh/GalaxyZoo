import numpy as N
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix

class Galaxy(dense_design_matrix.DenseDesignMatrix):

    def __init__(self,which_set = 'train', axes=('b', 0, 1, 'c'), preprocessor = None):
        dtype = 'uint8'
        ntrain = 60000
        nvalidation = 1578
        self.img_shape = (3, 100, 100)
        self.n_classes = 37
        self.axes = axes;
        self.img_size = N.prod(self.img_shape)
        
        fdata_names = ['galaxy_batch_%i.npy' % i for i in range(0,7)]
        flabels_names = ['galaxy_batch_solution_%i.npy' % i for i in range(0,7)]
        lenx = N.ceil((ntrain) / 10000.)*10000
        x = N.zeros((lenx,self.img_size), dtype=dtype)
        y = N.zeros((lenx, self.n_classes), dtype=dtype)
        
        nloaded = 0
        for i in xrange(len(fdata_names)):
            data = N.load(fdata_names[i])
            labels = N.load(flabels_names[i]) 
            x[i*10000:(i+1)*10000, :] = data 
            y[i*10000:(i+1)*10000] = labels
            nloaded += len(data)
            #if nloaded >= ntrain + nvalidation : break;
        
        Xs = {
                  'train' : x[0:ntrain],
                  'validation' : x[ntrain:] 
             }
        
        Ys = {
                  'train' : y[0:ntrain],
                  'validation' : y[ntrain:]
             }
        X = Xs[which_set]
        y = Ys[which_set]
  
        if isinstance(y, list):
            y = N.asarray(y)
            
        if which_set == 'validation':
            print y
            assert y.shape[0] == 1578
        
        view_converter = dense_design_matrix.DefaultViewConverter((100, 100, 3), axes)
        super(Galaxy, self).__init__(X=X, y=y, view_converter=view_converter)
        
        if preprocessor:
            proprocessor.apply(self)
