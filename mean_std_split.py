from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import os
import torch
from opts import *
from tqdm import tqdm

def get_score_mean_std(annotations):
    scores = []
    for a in annotations:
        score = a[1]
        scores.extend([score])
    s = np.asarray(scores)
    mean = np.mean(s)
    std = np.std(s)

    print('score mean:', mean)
    print('score stddev:', std)

    return mean, std

def get_video_mean_std(dataloader):
    print("calculating video mean, std")

    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data in tqdm(dataloader, leave=False, disable=disable_tqdm):
        video = data['video'].transpose_(1, 2).cuda()
        # batch_size, C, frames, H, W = video.shape

        channels_sum += torch.mean(video, dim=[0, 2, 3, 4])
        channels_sqrd_sum += torch.mean(video ** 2, dim=[0, 2, 3, 4])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    print('video mean:', mean)
    print('video std:', std)

    return mean.data.cpu().numpy(), std.data.cpu().numpy()


def make_split():
    scores = loadmat('../AQA-7-dataset/trampoline/trampoline_scores.mat').get('scores')
    sample_nos = np.vstack(np.arange(1, len(scores)+1))
    score_sample = np.append(sample_nos, scores, 1)

    np.random.shuffle(score_sample)
    train_list = score_sample[:65]
    test_list = score_sample[65:]

    train_dic = {'train_samples': train_list}
    test_dic = {'test_samples': test_list}
    if not os.path.exists(os.path.join(root_dir, 'trampoline_train.mat')):
        print("new split")
        savemat(os.path.join(root_dir, 'trampoline_train.mat'), train_dic)
        savemat(os.path.join(root_dir, 'trampoline_test.mat'), test_dic)

def print_split_info(rd=root_dir):
    train = loadmat(os.path.join(rd, 'trampoline_train.mat')).get('train_samples')
    print(*train)
    print("train:", np.mean(train, axis=0)[1], np.std(train, axis=0)[1])

    test = loadmat(os.path.join(rd, 'trampoline_test.mat')).get('test_samples')
    print(*test)
    print("test:", np.mean(test, axis=0)[1], np.std(test, axis=0)[1])