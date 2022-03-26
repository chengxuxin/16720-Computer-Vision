import imp
import numpy as np
from LucasKanadeAffine import *
import scipy
import scipy.ndimage.morphology as morphology
from InverseCompositionAffine import *
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    M = np.linalg.inv(M)
    image1_warped = scipy.ndimage.affine_transform(image2, M, output_shape = image1.shape)

    sub = np.abs(image1 - image1_warped)
    moving_pts = np.where((sub > tolerance) & (image1 != 0.) & (image1_warped != 0.))
    # moving_pts = np.where((sub / image1 > 1) & (image1 != 0.) & (image1_warped != 0.))
    mask[moving_pts] = 0

    mask = morphology.binary_erosion(mask, structure = np.ones((2,2)))
    mask = morphology.binary_dilation(mask)
    return mask
