import numpy as N
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix

class Galaxy(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set='train', axes=('b', 0, 1, 'c'), preprocessor=None):
        dtype = 'uint8'
        total_imgs = 61578
        batch_size = 10263
        ntrain = 5 * batch_size
        nvalidation = batch_size
        img_dim = 20
        self.img_shape = (3, img_dim, img_dim)
        self.n_classes = 37
        self.axes = axes;
        self.img_size = N.prod(self.img_shape) / 3

        num_batches = total_imgs / batch_size
        fdata_names = ['data/galaxy_batch_20_%i_grey.npy' % i for i in range(0, num_batches)]
        flabels_names = ['data/galaxy_batch_solution_%i.npy' % i for i in range(0, num_batches)]

        x = N.zeros((batch_size * num_batches, self.img_size), dtype=dtype)
        y = N.zeros((batch_size * num_batches, self.n_classes), dtype=dtype)

        nloaded = 0
        for i in xrange(len(fdata_names)):
            data = N.load(fdata_names[i])
            labels = N.load(flabels_names[i])
            x[i * batch_size:(i + 1) * batch_size] = data
            y[i * batch_size:(i + 1) * batch_size] = labels
            nloaded += batch_size
            #if nloaded >= ntrain + nvalidation : break;

        Xs = {
            #'train': x[0:ntrain],
            'train': x[0:ntrain-batch_size],
            'validation': x[ntrain:],
            'test': x[ntrain-batch_size:ntrain]
        }

        Ys = {
            #'train': y[0:ntrain],
            'train': y[0:ntrain-batch_size],
            'validation': y[ntrain:],
            'test': y[ntrain-batch_size:ntrain]
        }
        X = N.cast['float32'](Xs[which_set])
        y = Ys[which_set]

        if isinstance(y, list):
            y = N.asarray(y)

        if which_set == 'validation':
            assert y.shape[0] == batch_size

        view_converter = dense_design_matrix.DefaultViewConverter((img_dim, img_dim, 3), axes)
        super(Galaxy, self).__init__(X=X, y=y, view_converter=view_converter)

        if preprocessor:
            proprocessor.apply(self)
