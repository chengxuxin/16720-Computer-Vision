import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import skimage.io
import sklearn.cluster
from skimage.util import img_as_float
from tqdm import tqdm

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)

    [hint]
    * To produce the expected collage, loop first over scales, then filter types, then color channel.
    * Note the order argument when using scipy.ndimage.gaussian_filter. 
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----

    # preprocessing
    img = img_as_float(img)
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    
    try:
        assert np.max(img) < 1.1 and np.min(img) > -0.1
    except:
        print(np.max(img), np.min(img))
    
    img = skimage.color.rgb2lab(img[:, :, :3])
    # apply filters
    F = 4 * len(filter_scales)
 
    filter_responses = np.zeros((*img.shape, len(filter_scales), 4))
    for scale_idx, scale in enumerate(filter_scales):
        for channel_idx in range(3):
            filter_responses[:, :, channel_idx, scale_idx, 0] = scipy.ndimage.gaussian_filter(img[:,:,channel_idx], sigma=scale) # Gaussian
            filter_responses[:, :, channel_idx, scale_idx, 1] = scipy.ndimage.gaussian_laplace(img[:,:,channel_idx], sigma=scale) # Gaussian Laplace
            filter_responses[:, :, channel_idx, scale_idx, 2] = scipy.ndimage.gaussian_filter(img[:,:,channel_idx], sigma=scale, order=(0,1)) # Gaussian Derivative y
            filter_responses[:, :, channel_idx, scale_idx, 3] = scipy.ndimage.gaussian_filter(img[:,:,channel_idx], sigma=scale, order=(1,0)) # Gaussian Derivative x
    
    return filter_responses.reshape((*filter_responses.shape[:2], -1))

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    opts, file_name = args
    # global opts
    
    img = skimage.io.imread(file_name)
    filter_responses = extract_filter_responses(opts, img)
    h, w, F = filter_responses.shape
    sample_responses = np.zeros((opts.alpha, F))
    sampled_coord_x = np.random.randint(0, h, opts.alpha)
    sampled_coord_y = np.random.randint(0, w, opts.alpha)
    sample_responses = filter_responses[sampled_coord_x, sampled_coord_y]
    return sample_responses

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    #  For testing purpose, you can create a train_files_small.txt to only load a few images.
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    get_abs_path = lambda path: os.path.join(data_dir, path)
    train_files = list(map(get_abs_path, train_files))
    n_files = len(train_files)
    
    # train_files = [train_files[i] for i in np.random.randint(n_files, size=100)] # random sample some files for debugging
    # n_files = len(train_files)

    # multiprocessing
    pool = multiprocessing.Pool(processes=n_worker)
    args = [(opts, img_path) for img_path in train_files]
    results = pool.map(compute_dictionary_one_image, args, chunksize=1) # list of [alpha, 3F]
    # results = tqdm(pool.imap_unordered(compute_dictionary_one_image, train_files), total=n_files) # list of [alpha, 3F]

    # single thread
    # results = []
    # for file in tqdm(train_files):
    #     results.append(compute_dictionary_one_image(opts, file))

    all_responses = np.vstack(list(results))
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(all_responses)
    dictionary = kmeans.cluster_centers_  # [K, channel*n_filters*n_scales]
    
    ## example code snippet to save the dictionary
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    response = extract_filter_responses(opts, img)
    flat_response = response.reshape((-1, response.shape[-1])) # [n_pixel, F]
    dists = scipy.spatial.distance.cdist(flat_response, dictionary, metric='euclidean')  # [n_pixel, F], [K, F] retrun [n_pixel, n_center]
    words_idx = np.argmin(dists, axis=1)
    word_map = words_idx.reshape(response.shape[:2])
    return word_map

