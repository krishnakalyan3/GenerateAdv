import cPickle as pickle
import os
import numpy as np
import urllib
import gzip


def load_MNIST_dataset(mode=None):
    # This function borrowed from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            print("Downloading Images %s" % filename)
            urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/' + filename, filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            print("Downloading Labels %s" % filename)
            urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/' + filename, filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

        # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    y_train_mat = np.zeros((y_train.shape[0], 10), dtype=np.uint8)
    y_train_mat[np.arange(len(y_train)), y_train] = 1
    y_val_mat = np.zeros((y_val.shape[0], 10), dtype=np.uint8)
    y_val_mat[np.arange(len(y_val)), y_val] = 1

    y_test_mat = np.zeros((y_test.shape[0], 10), dtype=np.uint8)
    y_test_mat[np.arange(len(y_test)), y_test] = 1

    if mode =='onehot':
        return X_train, y_train_mat, X_val, y_val_mat, X_test, y_test_mat
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test



def load_cifar_dataset(Normal_flag=False):

    cifar_dir = 'cifar-10-batches-py'
    if not os.path.isdir(cifar_dir):
        print("Downloading...")
        urllib.urlretrieve("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")
        print("Extracting Files")
        os.system("tar xzvf cifar-10-python.tar.gz")

    # Load training set
    labels = []
    all_data = [pickle.load(open('cifar-10-batches-py/data_batch_' + str(i + 1), 'rb')) for i in range(5)]
    imgs = np.vstack([data.get('data') for data in all_data])
    X = imgs.reshape(50000, 3, 32, 32)
    # convert pixel values to range [0,1]
    X = X / np.float32(256)
    for data in all_data:
        x = data.get('labels')
        labels.append(x)
    Y = (np.array(labels, dtype='uint8')).flatten()

    # Normalize training images
    if Normal_flag == True:
        mean_pixel = np.mean(X, axis=0)
        X -= mean_pixel
        print('Normalized training samples')
    X_train = X
    Y_train = Y

    # Load test set
    test_dic = pickle.load(open('cifar-10-batches-py/test_batch', 'rb'))
    X_test = test_dic.get('data')
    X_test = X_test.reshape(10000, 3, 32, 32)
    X_test = X_test / np.float32(256)
    y_test = test_dic.get('labels')
    y_test = (np.array(y_test, dtype='uint8')).flatten()

    if Normal_flag == True:
        X_test -= mean_pixel
        print('Normalized test samples based on training pixel mean\n')

    return X_train, Y_train, X_test, y_test, mean_pixel
