#code adopted from Paritosh Parmar (https://github.com/ParitoshParmar/C3D-LSTM--PyTorch)

import json
import os
import random
import warnings

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import DataLoader
from tqdm import tqdm

from mean_std_split import get_video_mean_std
from models.C3D import C3D
from models.I3D import I3D
from models.LSTM import LSTM
from opts import *
from tramp_dataset import TrampDataset


def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%03d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def extract_features(video, offset=0):
    _, _, frames, _, _ = video.shape
    clip_feats = torch.Tensor([]).cuda()

    for i in np.arange(10-offset, frames - clip_size + 1, config["stride"]):
        clip = video[:, :, i:i + clip_size, :, :]
        clip_feats_temp = model_feat(clip)
        if config["feat_extractor"] == "c3d" or config["feat_extractor"] == "r21d":
            clip_feats_temp.unsqueeze_(0)
            clip_feats_temp.transpose_(0, 1)
        elif config["feat_extractor"] == "i3d":
            clip_feats_temp.squeeze_(4)
            clip_feats_temp.squeeze_(3)
            clip_feats_temp.transpose_(1, 2)
        else:
            print("unknown feature extractor")

        clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)

    return clip_feats


def train_phase(optimizer, criterion, scheduler, epoch):
    model_feat.eval()
    model_lstm.train()

    losses = []
    pred_scores = []
    true_scores = []

    for s_copy in range(config["spatial_aug"]):
        for offset in range(config["temp_aug"]):
            loop = tqdm(train_dataloader, leave=False, disable=disable_tqdm)
            for data in loop:
                with torch.no_grad():
                    true_scores.extend(data['label_final_score'].data.numpy())
                    true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
                    video = data['video'].transpose_(1, 2).cuda()

                    clip_feats = extract_features(video, offset=offset)

                pred_final_score = model_lstm(clip_feats)
                pred_scores.extend([element[0] for element in pred_final_score.data.cpu().numpy()])

                loss = criterion(pred_final_score, true_final_score)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_description(
                    f"ep[{epoch}/{config['num_epochs']}], s_aug[{s_copy + 1}/{config['spatial_aug']}], t_aug[{offset + 1}/{config['temp_aug']}]")
                # loop.set_postfix(loss=loss.item(), lr=learning_rate)

    mean_loss = sum(losses) / len(losses)
    print(f"Mean loss: {mean_loss}")
    scheduler.step(mean_loss)

    rho, p = stats.spearmanr(pred_scores, true_scores)
    # print('[TRAIN] Predicted scores: ', pred_scores)
    # print('[TRAIN] True scores: ', true_scores)
    print('[TRAIN] Correlation: ', rho)

    mean_losses.extend([mean_loss])
    train_correlations.extend([rho])

def test_phase():
    with torch.no_grad():
        pred_scores = []
        true_scores = []

        model_feat.eval()
        model_lstm.eval()

        for data in tqdm(test_dataloader, leave=False, disable=disable_tqdm):
            true_scores.extend(data['label_final_score'].data.numpy())
            video = data['video'].transpose_(1, 2).cuda()

            clip_feats = extract_features(video)

            temp_final_score = model_lstm(clip_feats)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])

        rho, p = stats.spearmanr(pred_scores, true_scores)

        # print('[TEST] Predicted scores: ', pred_scores)
        # print('[TEST] True scores: ', true_scores)
        print('[TEST] Correlation: ', rho)

        test_correlations.extend([rho])


def fit_model():
    parameters_2_optimize = (list(model_lstm.parameters()))
    optimizer = optim.Adam(parameters_2_optimize, lr=config["learning_rate"])
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["lr_factor"], patience=config["lr_patience"],
                                                     verbose=True)

    # actual training, testing loops
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}-------------------------")

        train_phase(optimizer, criterion, scheduler, epoch)
        test_phase()

        # save model
        if epoch % config["model_ckpt_interval"] == 0:
            save_model(model_lstm, "model_lstm", epoch, os.path.join(root_dir, "saved_models"))

        # save stats
        split_info['results'] = {'mean_losses': mean_losses,
                                 'train_correlations': train_correlations,
                                 'test_correlations': test_correlations}


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    os.mkdir(root_dir)
    os.mkdir(os.path.join(root_dir, "saved_models"))

    if config['feat_extractor'] == "i3d":
        # loading the altered I3D (ie I3D upto before AvgPool) from file
        model_I3D = I3D()
        model_I3D.load_state_dict(torch.load('../pretrained_nets/rgb_imagenet.pt'))
        model_feat = model_I3D.cuda()
    elif config['feat_extractor'] == "c3d":
        # loading the altered C3D (ie C3D upto before fc-6) from file
        model_CNN_pretrained_dict = torch.load('../pretrained_nets/c3d.pickle')
        model_CNN = C3D()
        model_CNN_dict = model_CNN.state_dict()
        model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
        model_CNN_dict.update(model_CNN_pretrained_dict)
        model_CNN.load_state_dict(model_CNN_dict)
        model_feat = model_CNN.cuda()
    elif config['feat_extractor'] == "r21d":
        # load the R(2+1)D model from torch hub
        model_r21d = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)
        model_r21d.fc = nn.Identity()
        model_feat = model_r21d.cuda()

    # bring transforms into format that can be exported to JSON
    cfg = config.copy()
    cfg['transforms'] = str(cfg['transforms'])
    info = {'config': cfg}

    # make list: sample number, score
    scores = loadmat('../AQA-7-dataset/trampoline/trampoline_scores.mat').get('scores')
    sample_numbers = np.vstack(np.arange(1, len(scores) + 1))
    annotations = np.append(sample_numbers, scores, 1)

    for split in range(config['splits']):
        print(f"TRAINING MODEL \"{root_dir}\"")
        print(
            f"feat_extractor: {config['feat_extractor']}"
            f"\nstride: {config['stride']}"
            f"\ntemp_aug: {config['temp_aug']}"
            f"\nspatial_aug: {config['spatial_aug']}"
            f"\nsplits: {config['splits']}")

        print(f"\nSplit {split}--------------------------------------------------------------")
        split_info = {}

        # make split
        np.random.shuffle(annotations)
        train_list = annotations[:65]
        test_list = annotations[65:]

        # calculate training split mean and std
        train_dataset_pre = TrampDataset(train_list, "pre")
        train_dataloader_pre = DataLoader(train_dataset_pre, batch_size=1)
        vid_mean, vid_std = get_video_mean_std(train_dataloader_pre)
        score_mean, score_std = np.mean(train_list, axis=0)[1], np.std(train_list, axis=0)[1]

        split_info['train_split'] = {'samples': train_list[:, 0].tolist(), 'score_mean': score_mean,
                                     'score_std': score_std,
                                     'vid_mean': vid_mean.tolist(), 'vid_std': vid_std.tolist()}
        split_info['test_split'] = {'samples': test_list[:, 0].tolist(),
                                    'score_mean': np.mean(test_list, axis=0)[1],
                                    'score_std': np.std(test_list, axis=0)[1]}

        print("train:\n", *train_list)
        print("mean: ", np.mean(train_list, axis=0)[1], "std: ", np.std(train_list, axis=0)[1])
        print("test:\n", *test_list)
        print("mean: ", np.mean(test_list, axis=0)[1], "std: ", np.std(test_list, axis=0)[1])

        train_dataset = TrampDataset(train_list, "train", vid_mean, vid_std, score_mean, score_std)
        test_dataset = TrampDataset(test_list, "test", vid_mean, vid_std, score_mean, score_std)

        train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True,
                                      num_workers=num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

        model_lstm = LSTM(lstm_input_dim).cuda()
        # resume training:
        # model_lstm.load_state_dict(torch.load(os.path.join(root_dir, "saved_models/model_lstm_050.pth")))

        mean_losses = []
        train_correlations = []
        test_correlations = []

        fit_model()

        with open(os.path.join(root_dir, str('info-split-{:02d}.json'.format(split))), 'w') as si_file:
            json.dump(split_info, si_file, indent=2)
        info['split-' + str(split)] = split_info

    # write to JSON file
    with open(os.path.join(root_dir, 'info-full.json'), 'w') as file:
        json.dump(info, file, indent=2)
