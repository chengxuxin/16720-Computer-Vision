# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 27, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 
np.set_printoptions(precision=4, suppress=True)

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    u, s, vT = np.linalg.svd(I, full_matrices=False)
    s[3:] = 0.
    s = np.diag(s)
    LT = u @ s**(1/2)
    L = LT[:, :3].T

    B = s**(1/2) @ vT
    B = B[:3]
    return B, L

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    G = np.eye(3)
    G[-1, 0] = mu
    G[-1, 1] = nu
    G[-1, 2] = lam
    B_bas = np.linalg.inv(G).T @ B
    albedos, normals = estimateAlbedosNormals(B_bas)
    surface = estimateShape(normals, s)
    save_name = f'2f-mu{mu}-nu{nu}-lam{lam}.png'
    plotSurface(surface, save_name=save_name)

if __name__ == "__main__":
    I, L, s = loadData()

    B, L_est = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Part 2 (b)
    # Your code here
    plt.imsave('2b-a.png', albedoIm, cmap = 'gray')
    plt.imsave('2b-b.png', normalIm, cmap = 'rainbow')
    print(L)
    print(L_est)

    # Part 2 (d)
    # Your code here
    surface = estimateShape(normals, s)
    # plotSurface(surface, suffix='2d')


    # Part 2 (e)
    # Your code here
    B_enforce = enforceIntegrability(B, s)
    albedos_en, normals_en = estimateAlbedosNormals(B_enforce)
    surface_en = estimateShape(normals_en, s)
    # plotSurface(surface_en, suffix='2e')

    # Part 2 (f)
    # Your code here

    plotBasRelief(B_enforce, -3, 0, 1)
    plotBasRelief(B_enforce, 3, 0, 1)
    plotBasRelief(B_enforce, 0, 0, 1)
    plotBasRelief(B_enforce, 0, -3, 1)
    plotBasRelief(B_enforce, 0, 3, 1)
    plotBasRelief(B_enforce, 0, 0, 0.1)
    plotBasRelief(B_enforce, 0, 0, 3)





