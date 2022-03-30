import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence
import plotly.express as px
from plotly.offline import plot
from q5_ec_bundle_adjustment import *
# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    # ----- TODO -----
    # YOUR CODE HERE
    pts2 = []
    for i in range(temple_pts1.shape[0]):
        pt2 = epipolarCorrespondence(im1, im2, F, temple_pts1[i,0], temple_pts1[i,1])
        pts2.append(pt2)
    pts2 = np.vstack(pts2)
    print(temple_pts1.shape, pts2.shape)
    M2, C2, P = findM2(F, temple_pts1, pts2, intrinsics)
    return P, M2, C2



'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('../data/templeCoords.npz')
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE
    x1 = temple_coords_path["x1"]
    y1 = temple_coords_path["y1"]
    temple_pts1 = np.hstack((x1, y1))
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    # F = ransacF(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    P, M2, C2 = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)

    figure = px.scatter_3d(x = P[:,0], y = P[:,1], z = P[:,2])
    figure.update_traces(marker_size=4)
    # figure.update_layout(scene = dict(zaxis = dict(range =[3,4.2])))
    print(P.shape)
    figure.show()

    M1 = np.eye(4)[0:3, :]
    C1 = K1 @ M1
    np.savez("../output/q4_2.npz", F=F, M1=M1, M2=M2, C1=C1, C2=C2)