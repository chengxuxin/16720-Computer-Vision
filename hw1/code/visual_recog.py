import os, math, multiprocessing
from os.path import join
from copy import copy
from xml.sax.handler import feature_external_ges

import numpy as np
from PIL import Image

from visual_words import *


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist = np.bincount(wordmap.flatten(), minlength=K)
    hist = hist / np.sum(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    h, w = wordmap.shape
    for l in range(L, -1, -1):
        weight = 2**(l-L-1)
        if l == 0:
            weight *= 2
        n_cut = 2*l
        if l != 0:
            prop = np.arange(n_cut+1) / n_cut
        else:
            prop = np.array((0, 1))
        split_x = (prop * h).astype(int)
        split_y = (prop * w).astype(int)

        hists = np.zeros((K, n_cut, n_cut))
        if l == L:
            for nx in range(n_cut):
                for ny in range(n_cut):
                    hist = get_feature_from_wordmap(opts, wordmap[split_x[nx]:split_x[nx+1], split_y[ny]:split_y[ny+1]])
                    hists[:, nx, ny] = hist
            hist_all = weight * hists.reshape((K, -1))
        else:
            for nx in range(n_cut):
                for ny in range(n_cut):
                    hist = np.sum(old_hists[:, 2*nx:2*nx+2, 2*ny:2*ny+2], axis=(1, 2)) / 4
                    # hist /= np.sum(hist)
                    hists[:, nx, ny] = hist
            hist_all = np.hstack((hist_all, weight*hists.reshape((K, -1))))
        # print(np.sum(hists))
        old_hists = hists.copy()
    # print(hist_all.shape)
    # print(np.sum(hist_all))
    # print(hist_all)
    return hist_all.flatten()
    
def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    opts, img_path, dictionary = args
    img = skimage.io.imread(join(opts.data_dir,img_path))
    # print(np.max(img))
    # img = np.array(img).astype(np.float32)/255
    wordmap = get_visual_words(opts, img, dictionary)
    return get_feature_from_wordmap_SPM(opts, wordmap)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    get_abs_path = lambda path: os.path.join(data_dir, path)
    train_files = list(map(get_abs_path, train_files))
    n_files = len(train_files)

    # multiprocessing
    pool = multiprocessing.Pool(processes=n_worker)
    args = [(opts, img_path, dictionary) for img_path in train_files]
    results = pool.map(get_image_feature, args, chunksize=1)
    
    # single thread
    # results = []
    # for img_path in tqdm(train_files):
    #     results.append(get_image_feature((opts, img_path, dictionary)))

    features = np.stack(results, axis=0)
    ## example code snippet to save the learned system
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def similarity_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    scores = np.sum(np.minimum(word_hist, histograms), axis=1)
    # print(scores.shape)
    return scores

def predict_labels(args):
    opts, file, dictionary, features, train_labels = args
    feature = get_image_feature((opts, file, dictionary))
    scores = similarity_to_set(feature, features)
    # print(np.argmax(scores))
    return train_labels[np.argmax(scores)]
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    SPM_layer_num = trained_system['SPM_layer_num']
    true_labels = trained_system['labels']
    
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    # ----- TODO -----
    get_abs_path = lambda path: os.path.join(data_dir, path)
    test_files = list(map(get_abs_path, test_files))
    n_files = len(test_files)
    
    # multiprocessing
    pool = multiprocessing.Pool(processes=n_worker)
    args = [(opts, img_path, dictionary, features, true_labels) for img_path in test_files]
    # results = tqdm(pool.imap(predict_labels, args), total=n_files)
    results = pool.map(predict_labels, args, chunksize=1)
    pool.close()
    pool.join()

    # single thread
    # results = []
    # for img_path in tqdm(test_files):
    #     results.append(predict_labels((opts, img_path, dictionary, features, true_labels)))
    

    predicted_labels = np.array(results)
    print(predicted_labels.shape, test_labels.shape, np.max(test_labels), np.max(predicted_labels))
    conf = sklearn.metrics.confusion_matrix(test_labels, predicted_labels)
    accuracy = np.trace(conf) / np.sum(conf)
    print(conf.shape, np.sum(conf))
    print(accuracy)
    print(conf)
    return conf, accuracy