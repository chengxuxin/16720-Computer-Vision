import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = M[:2, :3].flatten()
    
    x1, y1, x2, y2 = 0, 0, It.shape[0] -1, It.shape[1] -1
    It_int = RectBivariateSpline(np.arange(It.shape[0]),
                                 np.arange(It.shape[1]), 
                                 It)
    It1_int = RectBivariateSpline(np.arange(It1.shape[0]),
                                 np.arange(It1.shape[1]), 
                                 It1)

    dp = np.repeat([np.inf], 6)
    for i in range(num_iters):
        if np.linalg.norm(dp) < threshold:
            break
        x = np.arange(x1, x2 + 1)
        y = np.arange(y1, y2 + 1)
        X, Y = np.meshgrid(x,y)

        warp_x = p[0]*X + p[1]*Y + p[2]
        warp_y = p[3]*X + p[4]*Y + p[5]
        mask = np.where((warp_x >= x1) & (warp_x <= x2) & (warp_y >= y1) & (warp_y <= y2), True, False)
        X = X[mask]
        Y = Y[mask]
        warp_x = warp_x[mask]
        warp_y = warp_y[mask]

        b = It_int.ev(X, Y) - It1_int.ev(warp_x, warp_y)
        b = b.flatten()

        dx = It1_int.ev(warp_x, warp_y, dx = 1, dy = 0).flatten()
        dy = It1_int.ev(warp_x, warp_y, dx = 0, dy = 1).flatten()
        A = np.array([dx*X, dx*Y, dx, dy*X, dy*Y, dy]).T

        dp = np.linalg.lstsq(A, b)[0]

        p += dp
    M = np.vstack((p.reshape(2, 3), M[[2]]))
    return M
