import sys
import argparse
import os
import gc
import pandas as pd
import numpy as np
import random
from math import floor, ceil
import copy

import cv2
from PIL import Image

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


import torch
from torchvision import datasets, models, transforms

import albumentations
from albumentations import pytorch as AT
from .Augmentation import *

from .DataSampler import *


############################################ Define augments for dataset

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19", \
    required=False, help='specify the path of train dataset')
parser.add_argument('--df_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/train.csv", \
    required=False, help='specify the df path of train dataset')
parser.add_argument('--test_data_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19", \
    required=False, help='specify the path data of test dataset')
parser.add_argument('--test_df_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/test.csv", \
    required=False, help='specify the df path of test dataset')
parser.add_argument('--n_splits', type=int, default=5, \
    required=False, help='specify the number of folds')
parser.add_argument('--split', type=str, default="StratifiedKFold", required=False, help="specify the splitting dataset way")
parser.add_argument('--seed', type=int, default=42, \
    required=False, help='specify the random seed for splitting dataset')
parser.add_argument('--save_path', type=str, default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
    required=False, help='specify the path for saving splitted csv')
parser.add_argument('--test_fold', type=int, default=0, \
    required=False, help='specify the test fold for testing dataloader')
parser.add_argument('--batch_size', type=int, default=4, \
    required=False, help='specify the batch_size for testing dataloader')
parser.add_argument('--val_batch_size', type=int, default=4, \
    required=False, help='specify the val_batch_size for testing dataloader')
parser.add_argument('--num_workers', type=int, default=0, \
    required=False, help='specify the num_workers for testing dataloader')

############################################ Define constant
IMAGE_HEIGHT, IMAGE_WIDTH = 137, 236
IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE = 128, 128

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=IMAGE_HEIGHT_RESIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 60)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < IMAGE_WIDTH - 13) else IMAGE_WIDTH
    ymax = ymax + 10 if (ymax < IMAGE_HEIGHT - 10) else IMAGE_HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


############################## Prapare Augmentation
train_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE),
    albumentations.OneOf([
        # albumentations.Cutout(num_holes=4, max_h_size=4, max_w_size=4, fill_value=0),
        albumentations.ShiftScaleRotate(scale_limit=.15, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT),
        albumentations.IAAAffine(shear=20, mode='constant'),
        albumentations.IAAPerspective(),
        # albumentations.GridDistortion(distort_limit=0.3), 
    ], p=0.8)
    ])


test_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE),
    ])



############################################ Define Dataset 
class bengaliai_Dataset(torch.utils.data.Dataset):
    def __init__(self, \
                data_path, \
                df, \
                mode='train', \
                labeled=True, \
                transform = transforms.Compose([transforms.RandomResizedCrop(128),transforms.ToTensor()]), \
                grapheme_root_labels_dict={}, \
                vowel_diacritic_labels_dict={}, \
                consonant_diacritic_labels_dict={}, \
                grapheme_labels_dict={}):

        self.data_path = data_path
        self.df = df
        self.mode = mode
        self.labeled = labeled
        self.transform = transform
        self.image_df = pd.concat([pd.read_parquet(os.path.join(data_path, f'train_image_data_{i}.parquet')) for i in range(4)]).reindex()
        
        if self.labeled:
            self.uid = self.df['image_id'].values
        else:
            # we don't need to spilt
            self.uid = self.image_df['image_id'].values
        self.images = self.image_df.drop('image_id', axis=1).values.astype(np.uint8)
    
        self.grapheme_root_labels_dict = grapheme_root_labels_dict
        self.vowel_diacritic_labels_dict = vowel_diacritic_labels_dict
        self.consonant_diacritic_labels_dict = consonant_diacritic_labels_dict
        self.grapheme_labels_dict = grapheme_labels_dict
        
        del self.image_df
        gc.collect()
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_id = self.uid[idx]
        
        if self.mode == "train" or self.mode == "val":
            image_idx = int(image_id[6:])
        else:
            image_idx = int(image_id[5:])
            
        # print(image_id, image_idx)
        image = self.images[image_idx].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        # image = \
        #     self.image_df.loc[self.image_df["image_id"] == image_id, self.image_df.columns[1:]].values.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            
        image = 255 - image
        # image = crop_resize(image)
        
        # if ((self.mode == 'train') and (np.random.uniform() < 0.5)):
        #     image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype('uint8')
        #     image = augment_and_mix(image, severity=1, width=1, depth=-1, alpha=1.)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = np.float32(image)
        
        # if (self.mode == 'train'):
        #     for op in np.random.choice([
        #         lambda image : do_identity(image),
        #         lambda image : do_random_projective(image, 0.3, p=1),
        #         lambda image : do_random_perspective(image, 0.3, p=1),
        #         lambda image : do_random_scale(image, 0.4, p=1),
        #         lambda image : do_random_rotate(image, 0.4, p=1),
        #         lambda image : do_random_shear_x(image, 0.5, p=1),
        #         lambda image : do_random_shear_y(image, 0.4, p=1),
        #         lambda image : do_random_stretch_x(image, 0.5, p=1),
        #         lambda image : do_random_stretch_y(image, 0.5, p=1),
        #         lambda image : do_random_grid_distortion(image, 0.1, p=1),
        #         lambda image : do_random_custom_distortion1(image, 0.1, p=1),
        #     ],1):
        #         image = op(image)


        #     for op in np.random.choice([
        #         lambda image : do_identity(image),
        #         lambda image : do_random_erode(image, 0.2),
        #         lambda image : do_random_dilate(image, 0.2),
        #         lambda image : do_random_sprinkle(image, 0.2),
        #         lambda image : do_random_line(image, 0.3),
        #     ],1):
        #         image = op(image)

        if not (self.transform is None):
            
            image = self.transform(image=np.float32(image))['image']
        
        image = np.repeat(np.expand_dims(image, axis=0), 3, axis=0).astype(np.float32)
        image = image / 255
        
        if self.labeled:
            
            return image, \
                self.grapheme_root_labels_dict[image_id], \
                self.vowel_diacritic_labels_dict[image_id], \
                self.consonant_diacritic_labels_dict[image_id], \
                self.grapheme_labels_dict[image_id]
        else:
            return image



############################################ Define get functions
def get_train_val_split(df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/train.csv", \
                        save_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        n_splits=5, \
                        seed=42, \
                        split="MultilabelStratifiedKFold"):

    os.makedirs(save_path + 'split/' + split, exist_ok=True)
    df = pd.read_csv(df_path, encoding='utf8')
    df = df.fillna(0)
    
    # df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    
    if split == "MultilabelStratifiedKFold":
        kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed).split(df, \
                df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)
    elif split == "StratifiedKFold":
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed).split(df, \
                df[['grapheme']].values)
    elif split == "GroupKFold":
        # df = shuffle(df, random_state=seed)
        kf = GroupKFold(n_splits=n_splits).split(df, groups=df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)
        
    for fold, (train_idx, valid_idx) in enumerate(kf):
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(save_path + 'split/' + split + '/train_fold_%s_seed_%s.csv'%(fold, seed))
        df_val.to_csv(save_path + 'split/' + split + '/val_fold_%s_seed_%s.csv'%(fold, seed))

    return 


def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/",
                    df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/test.csv", \
                    batch_size=4, \
                    test_transform=test_transform):
    
    test_df = pd.read_csv(df_path)
    test_df = test_df.fillna(0)
    
    ds_test = bengaliai_Dataset(data_path, test_df, labeled=False, transform=test_transform)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    loader.num = len(test_df)
    
    return loader
    
    

def get_train_val_loaders(data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        train_df="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/split/train_fold_0_seed_42.csv", \
                        val_data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        val_df="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/split/train_fold_0_seed_42.csv", \
                        batch_size=4, \
                        val_batch_size=4, \
                        num_workers=2, \
                        train_transform=train_transform, \
                        val_transform=test_transform, \
                        Balanced="None"):
    

    
    train_df = pd.read_csv(train_df, encoding='utf8')
    val_df = pd.read_csv(val_df, encoding='utf8')
    df = pd.concat([train_df, val_df], axis=0)
    
    print(train_df.shape)
    print(val_df.shape)
    
    # print(train_df.columns)
    
    def prepare_labels(y):
        values = np.array(y)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        onehot_encoded = onehot_encoder.fit_transform(values.reshape(values.shape[0], 1))
        y = onehot_encoded
        return y
    
    class_map = dict(pd.read_csv(data_path + '/grapheme_1295.csv')[['grapheme','label']].values)
    train_df['grapheme'] = train_df['grapheme'].map(class_map)
    val_df['grapheme'] = val_df['grapheme'].map(class_map)
    df['grapheme'] = df['grapheme'].map(class_map)
    
    labels_encoded_grapheme_root = prepare_labels(df['grapheme_root'].values)
    img_class_dict_grapheme_root = {k:v for k, v in zip(df['image_id'].values, labels_encoded_grapheme_root)}
    
    labels_encoded_vowel_diacritic = prepare_labels(df['vowel_diacritic'].values)
    img_class_dict_vowel_diacritic = {k:v for k, v in zip(df['image_id'].values, labels_encoded_vowel_diacritic)}
    
    labels_encoded_consonant_diacritic = prepare_labels(df['consonant_diacritic'].values)
    img_class_dict_consonant_diacritic = {k:v for k, v in zip(df['image_id'].values, labels_encoded_consonant_diacritic)}
    
    labels_encoded_grapheme = prepare_labels(df['grapheme'].values)
    img_class_dict_grapheme = {k:v for k, v in zip(df['image_id'].values, labels_encoded_grapheme)}
    
    ds_train = bengaliai_Dataset(data_path, \
                                train_df, \
                                mode='train', \
                                labeled=True, \
                                transform=train_transform, \
                                grapheme_root_labels_dict=img_class_dict_grapheme_root, \
                                vowel_diacritic_labels_dict=img_class_dict_vowel_diacritic, \
                                consonant_diacritic_labels_dict=img_class_dict_consonant_diacritic, \
                                grapheme_labels_dict=img_class_dict_grapheme)
    
    if Balanced == "BalanceSampler":
        train_loader = torch.utils.data.DataLoader(ds_train, sampler=BalanceSampler(ds_train), \
        batch_size=batch_size, num_workers=num_workers, drop_last=True)
    elif Balanced == "ImbalancedDatasetSampler":
        train_loader = torch.utils.data.DataLoader(ds_train, sampler=ImbalancedDatasetSampler(ds_train), \
        batch_size=batch_size, num_workers=num_workers, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(ds_train, sampler=torch.utils.data.RandomSampler(ds_train), \
        batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    train_loader.num = len(train_df)

    ds_val = bengaliai_Dataset(data_path, \
                               val_df, \
                               mode='val', \
                               labeled=True, \
                               transform=val_transform, \
                               grapheme_root_labels_dict=img_class_dict_grapheme_root, \
                               vowel_diacritic_labels_dict=img_class_dict_vowel_diacritic, \
                               consonant_diacritic_labels_dict=img_class_dict_consonant_diacritic, \
                               grapheme_labels_dict=img_class_dict_grapheme)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    val_loader.num = len(val_df)

    return train_loader, val_loader

############################################ Define test function

def test_train_val_split(df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/train.csv", \
                        save_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        n_splits=5, \
                        seed=42, \
                        split="MultilabelStratifiedKFold"):
    
    print("------------------------testing train test splitting----------------------")
    print("df_path: ", df_path)
    print("save_path: ", save_path)
    print("n_splits: ", n_splits)
    print("seed: ", seed)

    get_train_val_split(df_path, save_path, n_splits, seed, split)

    print("generating successfully, please check results !")

    return


def test_train_val_loaders(data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        train_df="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/split/train_fold_0_seed_42.csv", \
                        val_data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        val_df="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/split/train_fold_0_seed_42.csv", \
                        batch_size=4, \
                        val_batch_size=4, \
                        num_workers=2, \
                        train_transform=train_transform, \
                        val_transform=test_transform):
    
    train_loader, val_loader = get_train_val_loaders(data_path=data_path, \
                     train_df=train_df, \
                     val_data_path=val_data_path, \
                     val_df=val_df, \
                     batch_size=batch_size, \
                     val_batch_size=val_batch_size, \
                     num_workers=num_workers, \
                     train_transform=train_transform, \
                     val_transform=val_transform)
        
    

    for image, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme in train_loader:
        print("------------------------testing train loader----------------------")
        print("image shape:", image.shape)
        print("grapheme_root shape: ", grapheme_root.shape)
        print("vowel_diacritic shape: ", vowel_diacritic.shape)
        print("consonant_diacritic shape: ", consonant_diacritic.shape)
        print("grapheme shape: ", grapheme.shape)
        print("image:", image)
        print("grapheme_root: ", grapheme_root)
        print("vowel_diacritic: ", vowel_diacritic)
        print("consonant_diacritic: ", consonant_diacritic)
        print("grapheme: ", grapheme)
        print("------------------------testing train loader finished----------------------")
        break

    for image, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme in val_loader:
        print("------------------------testing val loader----------------------")
        print("image shape:", image.shape)
        print("grapheme_root shape: ", grapheme_root.shape)
        print("vowel_diacritic shape: ", vowel_diacritic.shape)
        print("consonant_diacritic shape: ", consonant_diacritic.shape)
        print("grapheme shape: ", grapheme.shape)
        print("image:", image)
        print("grapheme_root: ", grapheme_root)
        print("vowel_diacritic: ", vowel_diacritic)
        print("consonant_diacritic: ", consonant_diacritic)
        print("grapheme: ", grapheme)
        print("------------------------testing val loader finished----------------------")
        break


def test_test_loader(data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/",
                    df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/test.csv", \
                    batch_size=4, \
                    test_transform=test_transform):

    test_loader = get_test_loader(data_path=data_path, \
                            df_path=df_path, \
                            batch_size=batch_size, \
                            test_transform=test_transform)

    for image in test_loader:
        print("------------------------testing test loader----------------------")
        print("image shape: ", image.shape)
        print("image: ", image)
        print("------------------------testing test loader finished----------------------")
        break


if __name__ == "__main__":

    args = parser.parse_args()
    
    # test split for training
    test_train_val_split(df_path=args.df_path, \
                        save_path=args.save_path, \
                        n_splits=args.n_splits, \
                        seed=args.seed, \
                        split=args.split)
    
     
    data_df_train = args.save_path + 'split/' + args.split + '/train_fold_%s_seed_%s.csv'%(args.test_fold, args.seed)
    data_df_val = args.save_path + 'split/' + args.split + '/val_fold_%s_seed_%s.csv'%(args.test_fold, args.seed)

    # test train val dataloader
    test_train_val_loaders(data_path=args.data_path, \
                        train_df=data_df_train, \
                        val_data_path=args.data_path, \
                        val_df=data_df_val, \
                        batch_size=args.batch_size, \
                        val_batch_size=args.val_batch_size, \
                        num_workers=args.num_workers, \
                        train_transform=train_transform, \
                        val_transform=test_transform)
    
    
    # test test dataloader
    test_test_loader(data_path=args.test_data_path,
                    df_path=args.test_df_path, \
                    batch_size=args.batch_size, \
                    test_transform=test_transform)