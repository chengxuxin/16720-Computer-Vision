import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    y1, x1, y2, x2 = rect
    It_int = RectBivariateSpline(np.arange(It.shape[0]),
                                 np.arange(It.shape[1]), 
                                 It)
    It1_int = RectBivariateSpline(np.arange(It1.shape[0]),
                                 np.arange(It1.shape[1]), 
                                 It1)
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    x, y = np.meshgrid(x,y)
    dp = np.array([np.inf, np.inf])

    for i in range(num_iters):
        if np.linalg.norm(dp) < threshold:
            break
        b = It_int.ev(x, y) - It1_int.ev(x+p[0], y+p[1])
        b = b.flatten()
        
        dx = It1_int.ev(x+p[0], y+p[1], dx = 1, dy = 0).flatten()
        dy = It1_int.ev(x+p[0], y+p[1], dx = 0, dy = 1).flatten()
        A = np.array([dx, dy]).T

        dp = np.linalg.lstsq(A, b)[0]
        p += dp

    return p
