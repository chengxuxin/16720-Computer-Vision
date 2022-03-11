import numpy as np
import cv2, os
from python.matchPics import matchPics
from python.helper import plotMatches
from python.opts import get_opts
import matplotlib.pyplot as plt

def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """

    matches, locs1, locs2 = matchPics(image1, image2, opts)

    #display matched features
    fig = plotMatches(image1, image2, matches, locs1, locs2)
    
    dir = "../output"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    # fig.savefig(dir + "/sigma{:.2f}-ratio{:.2f}.png".format(opts.sigma, opts.ratio), bbox_inches='tight')
    
if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2)
