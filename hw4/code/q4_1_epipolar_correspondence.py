import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here
from scipy.ndimage import gaussian_filter

# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', MarkerSize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    a, b, c = F.dot(np.array([x1, y1, 1]))
    im1 = gaussian_filter(im1, sigma = 2)
    im2 = gaussian_filter(im2, sigma = 2)
    y2s = np.arange(im2.shape[0])
    x2s = (-(b*y2s + c)/a).astype(int)
    # inside_indices = np.logical_and(y2s >= 0, y2s < im2.shape[1])
    # x2s = x2s[inside_indices]
    # y2s = y2s[inside_indices]

    patch_size = 10
    patch = im1[y1-patch_size:y1+patch_size, x1-patch_size:x1+patch_size]
    # print("patch1: ", x1, y1, im1.shape)
    for i in range(x2s.shape[0]):
        x2 = x2s[i]
        y2 = y2s[i]
        if x2 > patch_size and x2 < im2.shape[1]-patch_size and y2 > patch_size and y2 < im2.shape[0]-patch_size:
            patch2 = im2[y2-patch_size:y2+patch_size, x2-patch_size:x2+patch_size]
            # print(patch.shape, patch2.shape, y2, im2.shape)
            diff = patch - patch2
            diff = np.linalg.norm(diff.flatten())
            if not "min_diff" in locals():
                min_diff = diff
                min_index = i
            elif diff < min_diff:
                min_diff = diff
                min_index = i
    return x2s[min_index], y2s[min_index]

if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    epipolarMatchGUI(im1, im2, F)
    
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)
    np.savez("../output/q4_1.npz", F=F, pts1=pts1, pts2=pts2)