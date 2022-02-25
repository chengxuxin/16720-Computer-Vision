import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from displayMatch import *
# Q2.1.6

def get_match_num(args):
        img, i, opts = args
        deg = i*10
        rot_img = scipy.ndimage.rotate(img, deg)
        # cv2.imshow("debug", rot_img)
        # cv2.waitKey(1)
        matches, locs1, locs2 = matchPics(img, rot_img, opts)
        num = matches.shape[0]

        if deg in [0, 60, 120, 180]:
            fig = plotMatches(img, rot_img, matches, locs1, locs2)
            fig.savefig("../output/2.1.6-rot{}.jpg".format(deg), bbox_inches='tight')
        return (deg, num, i)
    
def rotTest(opts):
    img = cv2.imread('../data/cv_cover.jpg')
    # Read the image and convert to grayscale, if necessary
    
    args = [(img, i, opts) for i in range(36)]
    p = Pool(processes=multiprocessing.cpu_count())
    angles = []
    nums = []
    for para in tqdm(p.imap_unordered(get_match_num , args), total=36):
        deg, num, i = para
        angles.append(deg)
        nums.append(num)
    p.close()
    p.join()
    # Display histogram
    print(angles, nums)
    plt.figure(dpi=200)
    plt.bar(angles, nums, width=10, align="center")
    plt.xlabel("Angle")
    plt.ylabel("Number of matches")
    plt.savefig("../output/2.1.6.jpg")
    # plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
