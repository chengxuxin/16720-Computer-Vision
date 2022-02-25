from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()
    n_cpu = util.get_num_CPU()
    print(opts)
    ## Q1.1
    # img_path = join(opts.data_dir, "aquarium/sun_aztvjgubyrgvirup.jpg")
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    ## Q1.2
    print("Computing dictionary...")
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    # train_files = open(join(opts.data_dir, 'train_files.txt')).read().splitlines()
    # n_files = len(train_files)
    # import os
    # get_abs_path = lambda path: os.path.join(opts.data_dir, path)
    # train_files = list(map(get_abs_path, train_files))
    # n = 3
    # train_files = [train_files[i] for i in np.random.randint(n_files, size=n)]
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # imgs = []
    # maps = []
    # for file in train_files:
    #     img = Image.open(file)
    #     wordmap = visual_words.get_visual_words(opts, img, dictionary)
    #     imgs.append(img)
    #     maps.append(wordmap)
    # fig, axes = plt.subplots(2, 3)
    # for i in range(n):
    #     axes[0, i].imshow(imgs[i])
    #     axes[1, i].imshow(maps[i], cmap="Set1")
    #     axes[0, i].axes.get_xaxis().set_ticks([])
    #     axes[0, i].axes.get_yaxis().set_ticks([])
    #     axes[1, i].axes.get_xaxis().set_ticks([])
    #     axes[1, i].axes.get_yaxis().set_ticks([])
    # plt.show()
    # fig.savefig(join(opts.out_dir, "q1.3.jpg"))

    # img_path = join(opts.data_dir, 'aquarium/sun_aairflxfskjrkepm.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)
    
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # # print(hist)
    # hist_dim = opts.K * (4**(opts.L+1) - 1) / 3
    # print(hist.shape)
    # plt.bar(np.arange(10), hist)
    # plt.show()

    ## Q2.1-2.4
    # Q2.1
    print("Building system...")
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    print("Evaluating...")
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    # print(opts)
    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
