import numpy as N
import os
from PIL import Image
from scipy.io import savemat
from scipy.misc import imread
import sys

batch_size = 10263

def pickle_images(data_root, greyscale=False, save_name='galaxy_batch_'):
    imgs = [img for img in os.listdir(data_root) if not img.startswith('.')]
    imgs_mat = None
    thumbnail_size = 100
    for i, filename in enumerate(imgs):
        filepath = os.path.join(data_root, filename)
        if i != 0 and i % batch_size == 0:
            name = save_name + str(thumbnail_size) + '_' + str(i / batch_size - 1)
            if greyscale:
                name = name + '_grey'
            N.save(name, imgs_mat)
            imgs_mat = None
        if imgs_mat is None:
            size = thumbnail_size ** 2
            if not greyscale:
                size = size * 3
            imgs_mat = N.ndarray(shape=(batch_size, size), dtype=N.uint8)
        im = Image.open(filepath)
        if greyscale:
            im = im.convert('L')
        im.thumbnail((thumbnail_size, thumbnail_size), Image.ANTIALIAS)
        imgs_mat[i % batch_size] = N.array(im).flatten('F')
        if i % 100 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

    name = save_name + str(thumbnail_size) + '_' + str(i / batch_size)
    if greyscale:
        name = name + '_grey'
    N.save(name, imgs_mat[:i % batch_size + 1])

def pickle_testing(data_root, greyscale=False):
    pickle_images(data_root, greyscale, 'galaxy_unknown_batch_')

def pickle_labels(file_dir):
    labels = N.genfromtxt(file_dir, skip_header=0, comments='#', delimiter=',')
    for i in xrange(labels.shape[0] / batch_size):
        N.save('galaxy_batch_solution_' + str(i), labels[i * batch_size:(i + 1) * batch_size, 1:])
    if labels.shape[0] % batch_size != 0:
        N.save('galaxy_batch_solution_' + str(labels.shape[0] / batch_size), labels[(batch_size * (labels.shape[0] / batch_size)):, 1:])

if __name__ == "__main__":
    if sys.argv[1] == 'img':
        pickle_images(str(sys.argv[2]))
    elif sys.argv[1] == 'img_grey':
        pickle_images(str(sys.argv[2]), True)
    elif sys.argv[1] == 'label':
        pickle_labels(str(sys.argv[2]))
    elif sys.argv[1] == 'test_img':
        pickle_testing(str(sys.argv[2]), True)
    else:
        print "Valid action include 'img' and 'label'."
