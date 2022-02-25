import imp
from json import load
import numpy as np
import cv2
from helper import *
from matchPics import *
from planarH import *
from tqdm import tqdm 
import multiprocessing
from multiprocessing import Pool
# Import necessary functions

# Q3.1


def process_frame(args):
    target_frames, src_frames, i, opts, cv_cover = args
    frame = target_frames[i]
    srcframe = src_frames[i]

    w, h = frame.shape[:2]
    matches, locs1, locs2 = matchPics(cv_cover, frame, opts)
    locs1_od = locs1[matches[:, 0]]
    locs2_od = locs2[matches[:, 1]]
    locs1_od = locs1_od[:, [1, 0]]
    locs2_od = locs2_od[:, [1, 0]]

    locs1_od = np.hstack((locs1_od, np.ones_like(locs1_od[:, [0]])))
    locs2_od = np.hstack((locs2_od, np.ones_like(locs2_od[:, [0]])))

    H, inliers, pairs = computeH_ransac(locs1_od, locs2_od, opts)

    srcframe = cv2.resize(srcframe, (cv_cover.shape[1],cv_cover.shape[0]), interpolation = cv2.INTER_AREA)
    # print(srcframe.shape, cv_cover.shape)
    # warped_img = cv2.warpPerspective(srcframe, H, dsize=(h, w))
    final_img = compositeH(H, srcframe, frame)
    return (final_img, i)

if __name__ == "__main__":
    opts = get_opts()


    cv_cover = cv2.imread('../data/cv_cover.jpg')
    hp_cover = cv2.imread('../data/hp_cover.jpg')


    print(cv_cover.shape)
    cv_cover_ratio = cv_cover.shape[1] / cv_cover.shape[0]
    target_frames = loadVid("../data/book.mov")
    src_frames = loadVid("../data/ar_source.mov")
    src_frames = src_frames[:, 45: -45]
    srcwid = src_frames.shape[2]
    srcht = src_frames.shape[1]
    src_crop_lower = srcwid // 2 - int(srcht * cv_cover_ratio // 2)
    src_crop_higher = srcwid // 2 + int(srcht * cv_cover_ratio // 2)
    print(src_crop_lower, src_crop_higher)
    src_frames = src_frames[:, :, src_crop_lower : src_crop_higher]

    n_target = target_frames.shape[0]
    n_src = src_frames.shape[0]
    n_output = np.min([n_target, n_src])
    output_frames = target_frames.copy()[:n_output]

    print("target frames: {}, src frames: {}".format(target_frames.shape, src_frames.shape))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('../output/ar.mp4', fourcc, 30.0, (640, 480))

    args = [(target_frames, src_frames, i, opts, cv_cover) for i in range(n_output)]
    p = Pool(processes=multiprocessing.cpu_count())
    for result, i in tqdm(p.imap_unordered(process_frame , args), total=n_output):
        output_frames[i] = result
        out.write(result)
    p.close()
    p.join()

    out.release() 






# for i in tqdm(range(0, n_output, 100)):
#     frame = target_frames[i]
#     srcframe = src_frames[i]

#     w, h = frame.shape[:2]
#     matches, locs1, locs2 = matchPics(cv_cover, frame, opts)
#     locs1_od = locs1[matches[:, 0]]
#     locs2_od = locs2[matches[:, 1]]
#     locs1_od = locs1_od[:, [1, 0]]
#     locs2_od = locs2_od[:, [1, 0]]

#     locs1_od = np.hstack((locs1_od, np.ones_like(locs1_od[:, [0]])))
#     locs2_od = np.hstack((locs2_od, np.ones_like(locs2_od[:, [0]])))

#     H, inliers, pairs = computeH_ransac(locs1_od, locs2_od, opts)

#     srcframe = cv2.resize(srcframe, (cv_cover.shape[1],cv_cover.shape[0]), interpolation = cv2.INTER_AREA)
#     print(srcframe.shape, cv_cover.shape)
#     # warped_img = cv2.warpPerspective(srcframe, H, dsize=(h, w))
#     final_img = compositeH(H, srcframe, frame)

#     output_frames[i] = final_img
    # cv2.imwrite("../output/3.1-{}.jpg".format(i), final_img)
    # cv2.imwrite("../output/3.1-{}-src.jpg".format(i), srcframe)
    
    # if i == 3:
    #     break
    # plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
