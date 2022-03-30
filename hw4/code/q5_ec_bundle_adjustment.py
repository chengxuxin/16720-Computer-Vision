import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix
# from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=300, tol=10, return_inliers=False):
    # Replace pass by your implementation
    pts1_homo = toHomogenous(pts1)
    pts2_homo = toHomogenous(pts2)

    for i in range(nIters):
        indices = np.random.choice(pts1.shape[0], 8, replace=False)
        # Choose 8 random points
        pts1_sampled = pts1[indices]
        pts2_sampled = pts2[indices]
        # Estimate F
        F = eightpoint(pts1_sampled, pts2_sampled, M)
        # Calculate inliers
        error = calc_epi_error(pts1_homo, pts2_homo, F)
        inliers_count = np.sum(error < tol)
        # Update best F
        if not "max_inliers" in locals():
            max_inliers = inliers_count
            best_F = F
            inliers_indices = (error < tol)
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_F = F
            inliers_indices = (error < tol)
    if return_inliers:
        return best_F, inliers_indices
    return best_F


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def skew(omega):
    wx, wy, wz = omega
    w_hat = np.array([[0, -wz, wy],
                        [wz, 0, -wx],
                        [-wy, wx, 0]])
    return w_hat

def rodrigues(r):
    theta = np.linalg.norm(r)
    r = r / theta
    R = np.eye(3) + np.sin(theta) * skew(r) + (1 - np.cos(theta)) * np.dot(skew(r), skew(r))
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    # https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    A = (R - R.T)/2
    p = np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])
    s = np.linalg.norm(p)
    c = (np.trace(R) - 1) / 2

    if np.isclose(s, 0) and np.isclose(c, 1):
        return np.zeros(3)
    if np.isclose(s, 0) and np.isclose(c, -1):
        for i in range(3):
            if not np.isclose(np.sum(R[:, i]), -1):
                v = R[:, i]
                break
        u = v / np.linalg.norm(v)
        r = u * theta
        if np.sqrt(np.sum(r**2)) == np.pi and ((r[0, 0] == 0. and r[1, 0] == 0. and r[2, 0] < 0) or (r[0, 0] == 0. and r[1, 0] < 0) or (r[0, 0] < 0)):
            r = -r
        return r
    else:
        u = p / s
        theta = np.arctan2(s, c)
        r = u * theta
        return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    P = x[:-6]
    r2 = x[-6:-3]
    t2 = x[-3:]
    P, r2, t2 = x[:-6], x[-6:-3], x[-3:]
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, np.reshape(t2, (3, 1))))
    P = np.reshape(P, (-1, 3))
    P = np.vstack((P.T, np.ones((1, P.shape[0]))))
    p1_pred = K1 @ M1 @ P
    p1_pred = (p1_pred[:2, :] / p1_pred[2, :]).T
    p2_pred = K2 @ M2 @ P
    p2_pred  = (p2_pred[:2, :] / p2_pred[2, :]).T
    residuals = np.concatenate([(p1-p1_pred).reshape([-1]), (p2-p2_pred).reshape([-1])])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R2_0 = M2_init[:, :3]
    t2_0 = M2_init[:, 3]
    r2_0 = invRodrigues(R2_0)
    x = np.concatenate([P_init.flatten(), r2_0.flatten(), t2_0])
    # print(x.shape, P_init.shape, r2_0.shape, t2_0.shape)
    def obj(x):
        return np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x)**2)
    x_optimized = scipy.optimize.minimize(obj, x).x
    P = x_optimized[0:-6].reshape(-1, 3)
    r2 = x_optimized[-6:-3]
    t2 = x_optimized[-3:].reshape(3, 1)
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    # F, inliers_indices = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), return_inliers=True)
    # np.savez("../output/q5.npz", F=F, inliers_indices=inliers_indices)
    loaded = np.load('../output/q5.npz')
    F = loaded['F']
    inliers_indices = loaded['inliers_indices']

    # YOUR CODE HERE
    # F = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    # displayEpipolarF(im1, im2, F)
    # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    # assert(F.shape == (3, 3))
    # assert(F[2, 2] == 1)
    # assert(np.linalg.matrix_rank(F) == 2)
    

    # YOUR CODE HERE


    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)[:,0]) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)



    # YOUR CODE HERE
    p1 = noisy_pts1[inliers_indices]
    p2 = noisy_pts2[inliers_indices]

    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = K1 @ M1
    E = essentialMatrix(F, K1, K2)
    M2_all = camera2(E)

    C1 = np.dot(K1, M1)
    err_val = np.inf

    for i in range(M2_all.shape[2]):
        C2 = np.dot(K2, M2_all[:, :, i])
        w, err = triangulate(C1, p1, C2, p2)
        if err < err_val:
            err_val = err
            M2 = M2_all[:, :, i]
            C2_opt = C2
            w_best = w

    P_init, err = triangulate(C1, p1, C2_opt, p2)
    print(err)
    print(M2)
    print(w_best)


    M2_init, C2_init, P_init = findM2(F, p1, p2, intrinsics)
    
    M2, P, obj_start, obj_end = bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init[:, :3])

    C2 = K2 @ M2
    W = np.hstack((P, np.ones((P.shape[0], 1))))
    # print(W.shape)
    
    err = 0

    for i in range(p1.shape[0]):
        pj1 = C1 @ W[i].T
        pj2 = C2 @ W[i].T
        pj1 = (pj1[:2] / pj1[-1]).T
        pj2 = (pj2[:2] / pj2[-1]).T
        err += np.linalg.norm(pj1 - p1[i]) + np.linalg.norm(pj2 - p2[i])
    print(err)
    print(M2)
    
    plot_3D_dual(P_init, P)