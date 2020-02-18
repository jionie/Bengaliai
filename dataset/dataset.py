import sys
import argparse
import os
import gc
import pandas as pd
import numpy as np
import random
from math import floor, ceil

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


############################## Prapare Augmentation
train_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.RandomBrightness(0.15, p=1), 
        albumentations.RandomContrast(0.15, p=1),
        albumentations.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=1)
    ], p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, border_mode=1, p=0.5),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    AT.ToTensor(),
    ])

train_transform_advprop = albumentations.Compose([
    albumentations.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.RandomBrightness(0.15, p=1), 
        albumentations.RandomContrast(0.15, p=1),
        albumentations.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=1)
    ], p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, border_mode=1, p=0.5),
    albumentations.Lambda(lambda img: img * 2.0 - 1.0),
    AT.ToTensor(),
    ])


test_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    AT.ToTensor(),
    ])



############################################ Define Dataset 
class bengaliai_Dataset(torch.utils.data.Dataset):
    def __init__(self, \
                data_path, \
                df, \
                labeled=True, \
                transform = transforms.Compose([transforms.RandomResizedCrop(128),transforms.ToTensor()]), \
                grapheme_root_labels_dict={}, \
                vowel_diacritic_labels_dict={}, \
                consonant_diacritic_labels_dict={}, \
                grapheme_labels_dict={}):

        self.data_path = data_path
        self.df = df
        self.labeled = labeled
        self.transform = transform
        self.image_df = pd.concat([pd.read_parquet(os.path.join(data_path, f'train_image_data_{i}.parquet')) for i in range(4)]).reindex()
        self.uid = self.image_df['image_id'].values
        self.image = self.image_df.drop('image_id', axis=1).values.astype(np.uint8)
    
        self.grapheme_root_labels_dict = grapheme_root_labels_dict
        self.vowel_diacritic_labels_dict = vowel_diacritic_labels_dict
        self.consonant_diacritic_labels_dict = consonant_diacritic_labels_dict
        self.grapheme_labels_dict = grapheme_labels_dict
        #128547
        del self.image_df
        gc.collect()

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        
        image_id = self.uid[idx]
        image = self.image[idx].copy().reshape(137, 236)
        # image = image.astype(np.float32)/255
        
        if self.labeled:
            grapheme_root, vowel_diacritic, consonant_diacritic, grapheme = self.df.loc[self.df['image_id'] == image_id, \
                ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme']]
            
            return image, \
                self.grapheme_root_labels_dict[grapheme_root], \
                self.vowel_diacritic_labels_dict[vowel_diacritic], \
                self.consonant_diacritic_labels_dict[consonant_diacritic], \
                self.grapheme_root_labels_dict[grapheme]
        else:
            return image
    


############################################ Define get functions
def get_train_val_split(df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/train.csv", \
                        save_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/", \
                        n_splits=5, \
                        seed=42, \
                        split="MultilabelStratifiedKFold"):

    os.makedirs(save_path + '/split', exist_ok=True)
    df = pd.read_csv(df_path, encoding='utf8')
    df = df.fillna(0)
    
    # df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    
    kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(df, \
            df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)
        
    for fold, (train_idx, valid_idx) in enumerate(kf):
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(save_path + '/split/train_fold_%s_seed_%s.csv'%(fold, seed))
        df_val.to_csv(save_path + '/split/val_fold_%s_seed_%s.csv'%(fold, seed))

    return 


def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/",
                    df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/test.csv", \
                    batch_size=4, \
                    test_trainsform=test_transform):
    
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
                        val_transform=test_transform):
    

    
    train_df = pd.read_csv(train_df, encoding='utf8')
    val_df = pd.read_csv(val_df, encoding='utf8')
    
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
    
    labels_encoded_grapheme_root_train = prepare_labels(train_df['grapheme_root'].values)
    img_class_dict_grapheme_root_train = {k:v for k, v in zip(train_df['image_id'].values, labels_encoded_grapheme_root_train)}
    
    labels_encoded_vowel_diacritic_train = prepare_labels(train_df['vowel_diacritic'].values)
    img_class_dict_vowel_diacritic_train = {k:v for k, v in zip(train_df['image_id'].values, labels_encoded_vowel_diacritic_train)}
    
    labels_encoded_consonant_diacritic_train = prepare_labels(train_df['consonant_diacritic'].values)
    img_class_dict_consonant_diacritic_train = {k:v for k, v in zip(train_df['image_id'].values, labels_encoded_consonant_diacritic_train)}
    
    labels_encoded_grapheme_train = prepare_labels(train_df['grapheme'].values)
    img_class_dict_grapheme_train = {k:v for k, v in zip(train_df['image_id'].values, labels_encoded_grapheme_train)}
    
    ds_train = bengaliai_Dataset(data_path, \
                                train_df, \
                                labeled=True, \
                                transform=train_transform, \
                                grapheme_root_labels_dict=img_class_dict_grapheme_root_train, \
                                vowel_diacritic_labels_dict=img_class_dict_vowel_diacritic_train, \
                                consonant_diacritic_labels_dict=img_class_dict_consonant_diacritic_train, \
                                grapheme_labels_dict=img_class_dict_grapheme_train)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_loader.num = len(train_df)

    labels_encoded_grapheme_root_val = prepare_labels(val_df['grapheme_root'].values)
    img_class_dict_grapheme_root_val = {k:v for k, v in zip(val_df['image_id'].values, labels_encoded_grapheme_root_val)}
    
    labels_encoded_vowel_diacritic_val = prepare_labels(val_df['vowel_diacritic'].values)
    img_class_dict_vowel_diacritic_val = {k:v for k, v in zip(val_df['image_id'].values, labels_encoded_vowel_diacritic_val)}
    
    labels_encoded_consonant_diacritic_val = prepare_labels(val_df['consonant_diacritic'].values)
    img_class_dict_consonant_diacritic_val = {k:v for k, v in zip(val_df['image_id'].values, labels_encoded_consonant_diacritic_val)}
    
    labels_encoded_grapheme_val = prepare_labels(val_df['grapheme'].values)
    img_class_dict_grapheme_val = {k:v for k, v in zip(val_df['image_id'].values, labels_encoded_grapheme_val)}

    ds_val = bengaliai_Dataset(data_path, \
                               val_df, \
                               labeled=True, \
                               transform=val_transform, \
                               grapheme_root_labels_dict=img_class_dict_grapheme_root_val, \
                               vowel_diacritic_labels_dict=img_class_dict_vowel_diacritic_val, \
                               consonant_diacritic_labels_dict=img_class_dict_consonant_diacritic_val, \
                               grapheme_labels_dict=img_class_dict_grapheme_val)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    val_loader.df = val_df

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
        print("------------------------testing finetune train loader----------------------")
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
        print("------------------------testing finetune train loader finished----------------------")
        break

    for image, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme in val_loader:
        print("------------------------testing finetune val loader----------------------")
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
        print("------------------------testing finetune val loader finished----------------------")
        break


def test_test_loader(data_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/",
                    df_path="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/test.csv", \
                    batch_size=4, \
                    test_trainsform=test_transform):

    test_loader = get_test_loader(data_path=data_path, \
                            df_path=df_path, \
                            batch_size=batch_size, \
                            test_trainsform=test_trainsform)

    for image in test_loader:
        print("------------------------testing finetune test loader----------------------")
        print("image shape: ", image.shape)
        print("image: ", image)
        print("------------------------testing finetune test loader finished----------------------")
        break


if __name__ == "__main__":

    args = parser.parse_args()
    
    # test split for finetuning
    test_train_val_split(df_path=args.df_path, \
                        save_path=args.save_path, \
                        n_splits=args.n_splits, \
                        seed=args.seed, \
                        split="MultilabelStratifiedKFold")
    
     
    data_df_train = args.save_path + '/split/train_fold_%s_seed_%s.csv'%(args.test_fold, args.seed)
    data_df_val = args.save_path + '/split/val_fold_%s_seed_%s.csv'%(args.test_fold, args.seed)

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
                    test_trainsform=test_transform)