from matplotlib.pyplot import axis
import numpy as np
import cv2


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    A = []
    N = x1.shape[0]
    assert x1.shape == x2.shape
    
    for i in range(N):
        
        x1x = x1[i, 0]
        x1y = x1[i, 1]
        x2x = x2[i, 0]
        x2y = x2[i, 1]
        # A.append([-x1x, -x1y, -1, 0, 0, 0, x1x*x2x, x1y*x2x, x2x])
        # A.append([0, 0, 0, -x1x, -x1y, -1, x1x*x2y, x1y*x2y, x2y])
        A.append([-x2x, -x2y, -1, 0, 0, 0, x2x*x1x, x2y*x1x, x1x])
        A.append([0, 0, 0, -x2x, -x2y, -1, x2x*x1y, x2y*x1y, x1y])
        # if i == 0:
        #     print(x1[i], x2[i], A)
    U,S,Vt = np.linalg.svd(A)
    H2to1 = Vt[-1, :].reshape((3,3))
    H2to1 /= H2to1[2,2]
    return H2to1

def get_sim_mat(input):
    x = np.copy(input[:, :2])
    trans_mat = np.eye(3)
    
    mean = np.mean(x, axis=0)[None, :]
    # print(mean.shape)
    # x -= mean
    trans_mat[:2, -1] = -mean
    # print(trans_mat)
    # max_dis = np.max(np.linalg.norm(x[:, :2], ord=2, axis=1))
    max_dis = np.linalg.norm(x-mean, ord=2, axis=0)
    ratio = np.sqrt(2) / max_dis

    # x =  ratio[None, :] * x 
    # print(ratio)
    scale_mat = np.eye(3)
    scale_mat[0, 0] = ratio[0]
    scale_mat[1, 1] = ratio[1]
    T1 = scale_mat @ trans_mat
    return T1

def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points

    # Shift the origin of the points to the centroid

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)

    # Similarity transform 1
    T1 = get_sim_mat(x1)
    # Similarity transform 2
    T2 = get_sim_mat(x2)
    
    x1_norm = x1 @ T1.T
    x2_norm = x2 @ T2.T

    # Compute homography
    H2to1 = computeH(x1_norm, x2_norm)
    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
    return H2to1

def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol

    locs1 = locs1 / locs1[:, [-1]]
    locs2 = locs2 / locs2[:, [-1]]

    num_pts = locs1.shape[0]
    max_inliers = 0
    bestH = np.eye(3)
    the_inliers = None
    for i in range(max_iters):
        indices = np.random.randint(num_pts, size=4)
        chosen_locs1 = locs1[indices]
        chosen_locs2 = locs2[indices]
        H = computeH_norm(chosen_locs1, chosen_locs2)
        # H = computeH(chosen_locs1, chosen_locs2)

        pred_locs = locs2 @ H.T #H.dot(locs1)
        pred_locs /= pred_locs[:, [-1]]

        error = np.linalg.norm(pred_locs[:, :2] - locs1[:, :2], axis=1, ord=2)
        inliers = np.where(error<inlier_tol, 1, 0)
        num_inliers = np.sum(inliers)
        # print(num_inliers)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            bestH = H
            the_inliers = np.where(inliers==1)
            pairs = indices
    return bestH, np.array(the_inliers).squeeze(), pairs


def compositeH(H2to1, template, img):
    mask = np.ones_like(template)
    warped_mask = cv2.warpPerspective(mask, H2to1, dsize=(img.shape[1], img.shape[0]))

    warped_template = cv2.warpPerspective(template, H2to1, dsize=(img.shape[1], img.shape[0]))
    np.putmask(img, warped_mask==1, warped_template)

    return img
