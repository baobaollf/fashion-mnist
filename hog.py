# Linfeng Li
# Xing Qian
# CS 412
# University of Illinois at Chicago
# 11/20/2020

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def main():
    plt.gray()
    x_train, y_train = load_mnist('fashion-mnist/data/fashion', kind='train')
    x_test, y_test = load_mnist('fashion-mnist/data/fashion', kind='t10k')
    test_image = x_train[0]
    test_image = test_image.reshape(28, 28)
    plt.imshow(test_image)
    plt.show()


if __name__ == '__main__':
    main()