import numpy as np
import cv2, sys
import skimage.io 
import skimage.color
sys.path.append("../../")
from hw2.python.matchPics import matchPics
from opts import get_opts
from planarH import *
from displayMatch import *

# Import necessary functions

# Q2.2.4

def warpImage(opts):
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    w, h = cv_desk.shape[:2]
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    locs1_od = locs1[matches[:, 0]]
    locs2_od = locs2[matches[:, 1]]
    locs1_od = locs1_od[:, [1, 0]]
    locs2_od = locs2_od[:, [1, 0]]

    locs1_od = np.hstack((locs1_od, np.ones_like(locs1_od[:, [0]])))
    locs2_od = np.hstack((locs2_od, np.ones_like(locs2_od[:, [0]])))

    H, inliers, pairs = computeH_ransac(locs1_od, locs2_od, opts)
    print(inliers.shape)

    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]), interpolation = cv2.INTER_AREA)
    warped_img = cv2.warpPerspective(hp_cover, np.linalg.inv(H), dsize=(h, w))
    
    final_img = compositeH(np.linalg.inv(H), hp_cover, cv_desk)
    cv2.imwrite("../output/warp_test.jpg", final_img)

if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


