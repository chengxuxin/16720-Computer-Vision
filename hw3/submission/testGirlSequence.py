import argparse
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
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

height = rect[3] - rect[1]
width = rect[2] - rect[0]

rect_data = np.zeros((seq.shape[-1], 4))
rect_data[0] = rect
for frame_idx in range(seq.shape[-1]-1):
    It = seq[:, :, frame_idx]
    It1 = seq[:, :, frame_idx+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect = rect + np.array([p[1], p[0], p[1], p[0]])

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(It1, cmap = 'gray')
    patch = patches.Rectangle((rect[0],  rect[1]), width, height,
                                  linewidth = 1, edgecolor = 'r',
                                  facecolor = 'none')
    ax.add_patch(patch)
    
    fig.canvas.draw()
    # plt.pause(0.05)
    plt.close()
    
    if frame_idx in [1, 20, 40, 60, 80]:
        # cv2.imwrite("../output/q1.3-car_{}".format(frame_idx), )
        if not os.path.exists("../output"):
            os.makedirs("../output")
        fig.savefig("../output/q1.3-girl_{}.jpg".format(frame_idx), bbox_inches='tight')
    rect_data[frame_idx+1] = rect
np.savetxt("../output/girlseqrects.npy", rect_data)

plt.show()