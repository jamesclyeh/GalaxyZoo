import numpy as N
import os
from PIL import Image
from scipy.io import savemat
from scipy.misc import imread
import sys

batch_size = 10263

def pickle_images(data_root):
    imgs = os.listdir(data_root)
    imgs_mat = None
    thumbnail_size = 100
    for i, filename in enumerate(imgs):
        if not filename.startswith('.'):
            filepath = os.path.join(data_root, filename)
            if imgs_mat is None:
                imgs_mat = N.ndarray(shape=(batch_size, thumbnail_size ** 2 * 3), dtype=N.uint8)
            im = Image.open(filepath)
            im.thumbnail((thumbnail_size, thumbnail_size), Image.ANTIALIAS)
            imgs_mat[i % batch_size] = N.array(im).flatten('F')
            if i != 0 and i % batch_size == 0:
                N.save('galaxy_batch_' + str(i / batch_size - 1), imgs_mat)
                imgs_mat = None
        sys.stdout.write(".")
    #(fix) There is a .DS_Store file thus use filenum -1
    if (len(imgs) - 1) % batch_size != 0:
        N.save('galaxy_batch_' + str(i / batch_size), imgs_mat[:i % batch_size])

def pickle_labels(file_dir):
    labels = N.genfromtxt(file_dir, skip_header=0, comments='#', delimiter=',')
    for i in xrange(labels.shape[0] / batch_size):
        N.save('galaxy_batch_solution_' + str(i), labels[i * batch_size:(i + 1) * batch_size, 1:])
    if labels.shape[0] % batch_size != 0:
        N.save('galaxy_batch_solution_' + str(labels.shape[0] / batch_size), labels[(batch_size * (labels.shape[0] / batch_size)):, 1:])

if __name__ == "__main__":
    if sys.argv[1] == 'img':
        pickle_images(str(sys.argv[2]))
    elif sys.argv[2] == 'label':
        pickle_labels(str(sys.argv[2]))
    else:
        print "Valid action include 'img' and 'label'."
