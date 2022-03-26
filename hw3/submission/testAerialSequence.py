import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanadeAffine import *
from SubtractDominantMotion import *
from time import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.3, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
print(seq.shape)
start = time()

for frame_idx in range(seq.shape[-1] - 1):
    im1 = seq[: ,:, frame_idx]
    im2 = seq[:, :, frame_idx+1]

    mask = SubtractDominantMotion(im1, im2, threshold, num_iters, tolerance)
    pof = np.where(mask == 0)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(im1, cmap = 'gray')
    ax.plot(pof[1], pof[0],'.',color = 'r')
    if frame_idx in [1, 30, 60, 90, 120]:
        # fig.savefig('../output/q2.2-aerial-{}.png'.format(frame_idx), bbox_inches = "tight", pad_inches=0)
        fig.savefig('../output/q3.1-aerial-{}.png'.format(frame_idx), bbox_inches = "tight", pad_inches=0)
    plt.pause(0.001)
    plt.close()
print(time()- start)
