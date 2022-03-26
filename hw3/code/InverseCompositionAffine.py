import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
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
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)
    dx = It_int.ev(X, Y, dx = 1, dy = 0).flatten()
    dy = It_int.ev(X, Y, dx = 0, dy = 1).flatten()
    A = np.array([dx*X.flatten(), dx*Y.flatten(), dx, dy*X.flatten(), dy*Y.flatten(), dy]).T
    
    dp = np.repeat([np.inf], 6)
    for i in range(num_iters):
        if np.linalg.norm(dp) < threshold:
            break

        warp_x = p[0]*X + p[1]*Y + p[2]
        warp_y = p[3]*X + p[4]*Y + p[5]
        mask = np.where((warp_x >= x1) & (warp_x <= x2) & (warp_y >= y1) & (warp_y <= y2), True, False)
        # X_mask = X[mask]
        # Y_mask = Y[mask]
        # warp_x = warp_x[mask]
        # warp_y = warp_y[mask]

        b = -It_int.ev(X, Y) + It1_int.ev(warp_x, warp_y)
        b = b.flatten()
        b = np.where(mask.flatten(), b, 0)
        # Ap =  np.copy(A[mask.flatten()])

        dp = np.linalg.lstsq(A, b)[0]
        
        M = np.vstack((p.reshape(2,3), M[[2]]))
        dM = np.vstack((dp.reshape(2,3), M[[2]]))
        dM[0,0] += 1
        dM[1,1] += 1

        M = M @ np.linalg.inv(dM)
        p = M[0:2, 0:3].flatten()

    # M = np.vstack((p.reshape(2, 3), M[[2]]))

    return M
