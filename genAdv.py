#
"""
compatible with python version 2.7
@author: mahdieh.abbasi.1@ulaval.ca
"""

from __future__ import print_function
from __future__ import division
import os
import sys
import theano
import theano.tensor as T
import numpy as np
import lasagne
import cPickle as pickle
from scipy.optimize import fmin_l_bfgs_b as cnstOpt
import matplotlib.pyplot as plt
import getopt
import urllib


def load_cifar_dataset(Normal_flag = False):

    cifar_dir = 'cifar-10-batches-py'
    if not os.path.isdir(cifar_dir):
        print("Downloading...")
        urllib.urlretrieve("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")
        print("Extracting Files")
        os.system("tar xzvf cifar-10-python.tar.gz")


    # Load training set
    labels = []
    all_data=[pickle.load(open('cifar-10-batches-py/data_batch_'+str(i+1),'rb')) for i in range(5)]
    imgs = np.vstack([data.get('data') for data in all_data])
    X = imgs.reshape(50000,3,32,32)
    # convert pixel values to range [0,1]
    X = X/np.float32(256)
    for data in all_data:
        x = data.get('labels')
        labels.append(x)
    Y = (np.array(labels, dtype='uint8')).flatten()

    # Normalize training images
    if Normal_flag == True:
        mean_pixel = np.mean(X, axis=0)
        X-= mean_pixel
        print('Normalized training samples')
    X_train = X
    Y_train = Y

    # Load test set
    test_dic =pickle.load(open('cifar-10-batches-py/test_batch','rb'))
    X_test = test_dic.get('data')
    X_test=X_test.reshape(10000,3,32,32)
    X_test = X_test/ np.float32(256)
    y_test = test_dic.get('labels')
    y_test = (np.array(y_test, dtype='uint8')).flatten()

    if Normal_flag==True:
        X_test -= mean_pixel
        print('Normalized test samples based on training pixel mean\n')

    return X_train, Y_train, X_test, y_test, mean_pixel


def CudaConv_CNNcifar10(input_var=None, num_chan=3 ,width=32,num_fill = None, num_outputs=None):
    print('Architecture and hyper-parameters are selected from '+str('https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-18pct.cfg'))

    # Input Layer
    network = lasagne.layers.InputLayer(shape=(None, num_chan, width, width),
                                  input_var=input_var)


    network = lasagne.layers.Conv2DLayer(network, num_filters=num_fill[0], filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W = lasagne.init.HeNormal(gain='relu'),pad =2 , stride= 1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2 )
    network = lasagne.layers.LocalResponseNormalization2DLayer(network,n=3, alpha=5e-5)


    network = lasagne.layers.Conv2DLayer(network, num_filters=num_fill[1], filter_size=(5, 5), pad= 2 , stride=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
    network = lasagne.layers.Pool2DLayer(network, pool_size=(3,3), stride=2, pad=0,  mode='average_exc_pad')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network,n=3, alpha=5e-5)


    network = lasagne.layers.Conv2DLayer(network, num_filters=num_fill[2], filter_size=(5, 5), pad=2, stride=1,nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
    network = lasagne.layers.Pool2DLayer(network, pool_size=(3, 3), stride=2, pad=0, mode='average_exc_pad')

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p = 0.5),num_units=num_outputs ,nonlinearity=lasagne.nonlinearities.softmax)

    return network




def Load_weights(Pretrained_net, dataset_type):

    if dataset_type=='cifar10':
        print('load cifar10 dataset')
        size_images = 32
        num_channel = 3
        X_train, y_train, X_test, y_test, meanpixel = load_cifar_dataset(Normal_flag=True)
        num_classes = 10
        num_filter = [32, 32, 64]

    input_var = T.tensor4('inputs', dtype='float32')
    target_var = T.ivector('targets')

    # create CNN
    network = CudaConv_CNNcifar10(input_var=input_var, num_chan=num_channel, width=size_images, num_fill=num_filter, num_outputs=num_classes)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Theano functions
    val_fn_loss = theano.function([input_var, target_var], [test_loss], allow_input_downcast=True)
    val_fn_acc = theano.function([input_var, target_var], [test_acc], allow_input_downcast=True)
    grad = theano.grad(test_loss, input_var)
    fun_grad = theano.function([input_var, target_var], grad, allow_input_downcast=True)
    predic_fun = theano.function([input_var], [test_prediction], allow_input_downcast=True)

    print('load pretrained weights from '+str(Pretrained_net))
    # load the trained weights of CNN from a .pkl file
    net = pickle.load(open(Pretrained_net, 'rb'))
    # the loaded network (i.e. net) is a dictionary, where net['params'] is the trained weights of the loaded network
    all_param = net['params']
    lasagne.layers.set_all_param_values(network, all_param)
    net = {'loss': val_fn_loss, 'acc': val_fn_acc, 'predict': predic_fun, 'grad': fun_grad}
    return net, X_train, y_train, X_test, y_test, meanpixel


def objective(r, *args):
    net = args[0]
    base = args[1]
    orig_label = args[2]
    c = args[3]
    r = r.reshape(base.shape)
    r = np.asanyarray(r)
    test_loss = net['loss']
    grad_input = net['grad']
    # Frobenius norm is used [sum (a_ij)^2]^(1/2)
    obj = c * np.sqrt(np.sum(r ** 2)) + test_loss(base + r, orig_label)
    grad = grad_input(base + r, orig_label) + c * (r) / np.sqrt(np.sum(r ** 2))

    return obj, grad.flatten()


def LBFGS(net, X, y, Mean_pixel,  object_fun=objective, dataset_type=None):
    print('Generating adversarial examples by LBFGS')
    factr = 10.0
    pgtol = 1e-05

    distortion = []
    output_x = []
    outputVals = []
    output_y = []
    indx = []
    c = []
    p_softmax = net['predict']

    for i in range(X.shape[0]):

        print('sample {}'.format(i))
        base = X[i:i + 1].copy()
        orig_label = y[i:i + 1].copy()
        initial = np.ones(base.shape, dtype='float') * 1e-20
        lwr_bnd = -base - Mean_pixel
        upr_bnd =  - base +1-Mean_pixel
        bound = zip(lwr_bnd.flatten(), upr_bnd.flatten())


        # select a label != orig_label
        while True:
            fool_target = np.uint8(np.random.choice(range(10), 1))
            if fool_target != orig_label:
                print('Selected target {}, true target {}'.format(fool_target,orig_label))
                break
        print('Real label{}'.format(orig_label[0]))
        print('selected fool label{}'.format(fool_target))
        C = 3
        x, f, d = cnstOpt(object_fun, x0=initial.flatten().astype('float'),
                          args=(net, base, fool_target, C),
                          bounds=bound, maxiter=10000, iprint=0, factr=factr,
                          pgtol=pgtol)
        print('An adv example generated\n')
        print('prediction fool label {:.3f}'.format(p_softmax(x.reshape(base.shape) + base)[0][0,fool_target[0]]))
        print('prediction True label {:.3f}'.format(p_softmax(x.reshape(base.shape) + base)[0][0,orig_label[0]]))

        Ec_dist = np.sqrt(np.mean((x) ** 2))
        c.append(C)
        outputVals.append(p_softmax(x.reshape(base.shape) + base)[0])
        distortion.append(Ec_dist)
        print("Magnitude of distortion {:.6f}\n".format(Ec_dist))
        output_x.append(x.reshape(base.shape) + base)

        # for plot adversarial example and its corresponding clean sample, then their difference
        #
        """
        plt.subplot(3,1,1)
        plt.imshow((base[0]+Mean_pixel).transpose(1,2,0))
        plt.axis('off')
        plt.subplot(3, 1,2)
        plt.imshow(((x.reshape(base.shape) + base)[0]+Mean_pixel).transpose(1,2,0))
        plt.axis('off')
        plt.subplot(3, 1,3)
        plt.imshow((x.reshape(base.shape)[0]).transpose(1,2,0))
        plt.axis('off')
        plt.savefig(os.path.join('Images_cifar', 'adv'+str(i)+' C='+str('{:.2f}'.format(C))+'.jpg'))
        plt.close()
        """
        output_y.append(orig_label)
        indx.append(i)
        estimated_prediction = np.vstack(np.asarray(outputVals, dtype='float32'))
        print("Avg distortion {:.3f} AVG confidence {:.6f} \n".format(np.mean(distortion),
                                                                          np.mean(np.max(estimated_prediction,axis=1), axis=0)))
        print(np.max(p_softmax(x.reshape(base.shape) + base)[0]))
    data_fooled = zip(output_x, outputVals, output_y, distortion, indx, c)
    pickle.dump(data_fooled, open(dataset_type + '_'+str(X.shape[0])+'Adv_LBFGS_Test.pkl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


def FastSign( net_info, X, Y, dataset_type=None):
    print('Generating adversarial examples by Fast gradient sign method')
    fun_grad = net_info['grad']
    p_softmax = net_info['predict']
    output_x = []
    output_y = []
    distortion = []
    outputVals = []

    epsilon = 0.1

    for idx_init_img in range(len(X)):
        itc = 0
        prtb_x, prtb_y = X[idx_init_img:idx_init_img + 1].copy(), Y[idx_init_img:idx_init_img + 1].copy()
        orig_x = X[idx_init_img:idx_init_img + 1].copy()

        while itc < 200:
            itc += 1
            eta = epsilon * np.sign(fun_grad(prtb_x, prtb_y))
            prtb_x += eta
            fooled_y = np.uint8(np.argmax(p_softmax(prtb_x)[0], axis=1))
            if (prtb_y != fooled_y):
                print('fooled label {}'.format(fooled_y))
                print('True label {}'.format(prtb_y))
                Ec_dist = np.sqrt(np.mean((prtb_x[0] - orig_x[0]) ** 2))
                distortion.append(Ec_dist)
                outputVals.append(p_softmax(prtb_x)[0])
                output_x.append(prtb_x[0])
                output_y.append(prtb_y)
                estimated_prediction = np.vstack(np.asarray(outputVals, dtype='float32'))
                print("Avg distortion {:.4f}, Avg confidence {:.4f} ".format(np.mean(distortion), np.mean(np.max(estimated_prediction, axis=1),axis=0)))
                break

    data_fooled = zip(output_x, outputVals, output_y, distortion)

    pickle.dump(data_fooled, open(os.path.join(dataset_type+'_AdvExmple_Testsamples_' + str(len(X))+ '_FastSign.pkl'),'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return output_x, output_y, outputVals, distortion

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hm:d:n:", ["help", "method=","dataset=","PretrainedNet="])
    except getopt.GetoptError:
        raise("I dont understand input arguments")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h, --help"):
            print('Generating adversarial examples\n')
            print('usage: genAdv.py -m LBFGS -d cifar10 -n Cifar_pretrained_CudaConvVersion.pkl')
            print('-m or --method  takes the method for generating adversarial examples, option methods [LBFGS, FastSign]')
            print('-d, --dataset takes which dataset. For now just cifar10 is acceptable!')
            print('-n, --PretrainedNet takes a pkl filename, which contains trained weights of a net')
        elif opt in ("-m, --method"):
            print('Choosen method is '+str(arg))
            gener_method = arg
        elif opt in  ("-d, --dataset"):
            print('dataset is '+str(arg))
            data = arg
        elif opt in ("-n, --PretrainedNet"):
            print('pretrained net filename is '+str(arg))
            Pretrained_net = arg

    net, X_train, y_train, X_test, y_test, pixelmean = Load_weights(Pretrained_net, dataset_type=data)
    # Randomly select 100 samples from test set and save in a pkl file for reproduction purpose
    indx = np.random.choice(range(len(X_test)),100)
    pickle.dump(indx, open('index_1000_Selected_Testsamples.pkl','wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # indx = pickle.load(open('Selected_samples_index.pkl','rb'))
    X = X_test[indx]
    Y = y_test[indx]
    if gener_method =='LBFGS':
        LBFGS(net, X, Y, Mean_pixel=pixelmean,  object_fun=objective, dataset_type=data)
    elif gener_method == 'FastSign':
        FastSign(net, X,Y, dataset_type=data)

if __name__ == '__main__':
    main(sys.argv[1:])