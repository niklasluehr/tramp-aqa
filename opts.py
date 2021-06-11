#code adopted from Paritosh Parmar (https://github.com/ParitoshParmar/C3D-LSTM--PyTorch)

from torchvision.transforms import transforms

root_dir = '../06-11-final-i3d-temp_aug_8-3'
random_seed = 1
disable_tqdm = True

config = dict(
    feat_extractor='i3d',
    stride=16,
    temp_aug=8,
    spatial_aug=3,
    transforms=transforms.Compose([transforms.RandomApply([transforms.RandomAffine(0, shear=20)], p=0.42),
                                   transforms.RandomApply([transforms.RandomRotation(20, fill=0)], p=0.42)]),

    train_batch_size=5,
    test_batch_size=6,

    num_epochs=30,
    learning_rate=0.0001,
    lr_factor=0.1,
    lr_patience=5,
    splits=5,

    model_ckpt_interval=5,

    random_seed=random_seed
)

dataset_dir = '../AQA-7-dataset/trampoline'
sample_length = 618

if config['feat_extractor'] == "i3d":
    input_resize = 320, 240
    C, H, W = 3, 224, 224
    clip_size = 16
    lstm_input_dim = 1024
    num_workers = 1
elif config['feat_extractor'] == "c3d":
    # input_resize = 149,112
    input_resize = 171, 128
    C, H, W = 3, 112, 112
    clip_size = 16
    lstm_input_dim = 4096
    num_workers = 6
elif config['feat_extractor'] == "r21d":
    input_resize = 171, 128
    C, H, W = 3, 112, 112
    clip_size = 32
    lstm_input_dim = 512
    num_workers = 6
else:
    print("unknown feature extractor")
