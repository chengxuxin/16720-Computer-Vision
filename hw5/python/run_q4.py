import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import time

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    # plt.savefig("../output/q4.3-{}".format(img), dpi=200)
    # plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    sorted_boxes = []
    current_row = []
    bboxes.sort(key = lambda x:x[2])
    lower = bboxes[0][2]
    for box in bboxes:
        minr, minc, maxr, maxc = box
        if minr - lower > 50:
            sorted_boxes.append(current_row)
            current_row = []
            lower = maxr
        current_row.append(box)
    sorted_boxes.append(current_row)

    ####### test code ########
    # for row in sorted_boxes:
    #     print(len(row))
    #     row.sort(key = lambda x:x[1])
    #     for box in row:
    #         minr, minc, maxr, maxc = box
    #         im = bw[minr:maxr, minc:maxc]
    #         im = np.pad(im, ((50, 50), (50, 50)), 'constant', constant_values=1)
    #         plt.imshow(im, cmap='gray')
    #         plt.show()
    # raise

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    print(img)
    for row in sorted_boxes:
        this_line = ""
        row.sort(key = lambda x:x[1])
        for box in row:
            minr, minc, maxr, maxc = box
            im = bw[minr:maxr, minc:maxc]
            im = np.pad(im, 20*np.ones(4, dtype=int).reshape(2, 2), 'constant', constant_values=1)
            im = skimage.transform.resize(im, (32, 32), preserve_range=True)
            im = im.T.flatten().reshape(1, -1)
            # im[im>0.5] = 1.
            # im[im<=0.5] = 0.
            # print(im.max(), im.min(), im.shape)
            # plt.imshow(im.reshape(32, 32).T, cmap='gray')
            # plt.show()
            h1 = forward(im, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            preds = np.argmax(probs[0,:])
            this_line += (letters[preds] + "")
        print(this_line)
    #break