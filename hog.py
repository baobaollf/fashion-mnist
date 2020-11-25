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

import mnist_reader



def main():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


if __name__ == '__main__':
    main()