import argparse
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *
import cv2
import os
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=10000, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = np.array([59, 116, 145, 151])
rect0 = np.array(copy(rect))
rect_temp = np.array(copy(rect))
p0 = np.zeros(2)

height = rect[3] - rect[1]
width = rect[2] - rect[0]

rect_data = np.zeros((seq.shape[-1], 4))
rect_data[0] = rect

T1x = seq[:,:,0]
It = seq[:, :, 0]
for frame_idx in range(seq.shape[-1]-1):
    
    It1 = seq[:, :, frame_idx+1]
    p = LucasKanade(It, It1, rect_temp, threshold, num_iters, p0)

    pn = (rect_temp-rect0)[:2][::-1] + p

    pn_star = LucasKanade(T1x, It1, rect0, threshold, num_iters, p0=pn)
    if np.linalg.norm(pn_star - pn) < template_threshold:
        rect = rect0 + np.array([pn_star[1], pn_star[0], pn_star[1], pn_star[0]])
        
        It = seq[:, :, frame_idx+1]
        p0 = np.zeros(2)
        rect_temp = np.copy(rect)
    else:
        rect = rect_temp + np.array([p[1], p[0], p[1], p[0]])
        p0 = p
    

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(It1, cmap = 'gray')
    patch = patches.Rectangle((rect[0],  rect[1]), width, height, linewidth = 1, edgecolor = 'b',facecolor = 'none')
    ax.add_patch(patch)
    
    fig.canvas.draw()
    # plt.pause(0.05)
    plt.close()
    
    if frame_idx in [1, 100, 200, 300, 400]:
        # cv2.imwrite("../output/q1.3-car_{}".format(frame_idx), )
        if not os.path.exists("../output"):
            os.makedirs("../output")
        fig.savefig("../output/q1.3-car-wcrt_{}.jpg".format(frame_idx), bbox_inches='tight')
    rect_data[frame_idx+1] = rect
np.savetxt("../output/carseqrects-wcrt.npy", rect_data)

plt.show()