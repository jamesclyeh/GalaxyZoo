import numpy as N
import os
from PIL import Image
from scipy.io import savemat
from scipy.misc import imread
import sys

def pickle_images(data_root):
    batch_size = 10000
    # data_root = '/Users/jamesyeh/Downloads/images_training_rev1'
    imgs = os.listdir(data_root)
    imgs_mat = None
    thumbnail_size = 100
    for i, filename in enumerate(imgs):
        if not filename.startswith('.'):
            filepath = os.path.join(data_root, filename)
            if imgs_mat is None:
                imgs_mat = N.ndarray(shape=(batch_size, thumbnail_size * 3), dtype=N.uint8)
            im = Image.open(filepath)
            im.thumbnail((thumbnail_size, thumbnail_size), Image.ANTIALIAS)
            imgs_mat[i % batch_size] = N.array(im).flatten()
            if i != 0 and i % batch_size == 0:
                N.save('galaxy_batch_' + str(i / batch_size), imgs_mat)
                imgs_mat = None
        sys.stdout.write(".")
    N.save('galaxy_batch_' + str(i / batch_size + 1), imgs_mat[:i % batch_size])

if __name__ == "__main__":
    pickle_images(sys.argv[0])