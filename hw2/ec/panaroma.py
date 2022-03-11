import numpy as np
import cv2, sys
import skimage.io 
import skimage.color
sys.path.append("../")
from python.matchPics import matchPics
from python.opts import get_opts
from python.planarH import *
from python.displayMatch import *
from skimage.draw import polygon

# Import necessary functions

# Q2.2.4
def get_points(im, num_pts):
    print('Please select {} points in each image for alignment.'.format(num_pts))
    plt.imshow(im)
    pts = np.array(plt.ginput(num_pts, timeout=0))
    plt.close()
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def compositeH_pano(H2to1, template, img):
    offset = template.shape[1] // 2#430
    Trans = np.eye(3)
    Trans[0, -1] = offset
    H = Trans @ H2to1
    # shape = list(template.shape)
    # shape[1] = shape[1] + offset

    mask = np.ones_like(template)
    warped_mask = cv2.warpPerspective(mask, H, dsize=(img.shape[1]+offset, img.shape[0]))

    warped_template = cv2.warpPerspective(template, H, dsize=(img.shape[1]+offset, img.shape[0]))
    zeros = np.zeros((img.shape[0], offset, 3))
    print(img.shape, zeros.shape)

    img = np.hstack((zeros, img))
    np.putmask(img, warped_mask==1, warped_template)
    print(img.shape)
    img = img.astype(int)
    return img

def warpImage(opts):
    
    
    # pano_left = cv2.imread('../data/left.jpeg')
    # pano_right = cv2.imread('../data/right.jpeg')
    
    
    # np.savetxt("left_pts.txt", pts1)
    # np.savetxt("right_pts.txt", pts2)
    # pts1 = np.loadtxt("left_pts.txt")
    # pts2 = np.loadtxt("right_pts.txt")

    pano_left = cv2.resize(cv2.imread('../data/bay0.jpeg'), dsize=(800, 600))
    pano_right = cv2.resize(cv2.imread('../data/bay1.jpeg'), dsize=(800, 600))
    auto = True
    if auto == True:
        matches, locs1, locs2 = matchPics(pano_left, pano_right, opts)
        locs1_od = locs1[matches[:, 0]]
        locs2_od = locs2[matches[:, 1]]
        locs1_od = locs1_od[:, [1, 0]]
        locs2_od = locs2_od[:, [1, 0]]

        locs1_od = np.hstack((locs1_od, np.ones_like(locs1_od[:, [0]])))
        locs2_od = np.hstack((locs2_od, np.ones_like(locs2_od[:, [0]])))

        H, inliers, pairs = computeH_ransac(locs1_od, locs2_od, opts)
    else:
        
        pts1 = get_points(pano_left, 4)
        pts2 = get_points(pano_right, 4)
        H = computeH_norm(pts1, pts2)
    
    final_img = compositeH_pano(np.linalg.inv(H), pano_left, pano_right,)
    cv2.imwrite("../output/extra_left.png", pano_left)
    cv2.imwrite("../output/extra_right.png", pano_right)
    cv2.imwrite("../output/extra.png", final_img)
    
    plt.imshow(final_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


