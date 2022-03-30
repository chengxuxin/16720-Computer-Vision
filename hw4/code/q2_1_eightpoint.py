import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize
import os
# Insert your package here
np.set_printoptions(precision=4, suppress=True)


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1_norm = pts1 / M
    pts2_norm = pts2 / M
    x1 = pts1_norm[:,0]
    y1 = pts1_norm[:,1]
    x2 = pts2_norm[:,0]
    y2 = pts2_norm[:,1]
    
    U = np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones_like(x1))).T
    T = np.diag([1/M, 1/M, 1])
                  
    u, s, v = np.linalg.svd(U)
    F = v[-1, :].reshape(3,3)
    F = _singularize(F)
    F = refineF(F, pts1_norm, pts2_norm)
    
    F = T.T @ F @ T
    return F / F[2, 2]




if __name__ == "__main__":
        
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print(F)
    # Q2.1
    # Write your code here
    if not os.path.isdir('../output'):
        os.makedirs('../output')
    np.savez('../output/q2_1.npz', F=F, M=np.max([*im1.shape, *im2.shape]))
    displayEpipolarF(im1, im2, F)
    # plt.savefig('../output/q2.1.png')

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)