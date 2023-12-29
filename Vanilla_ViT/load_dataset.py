'''Module to load & transform dataset.
'''

import os
import glob
import cv2
from torch.utils.data import Dataset
import deeplake
import torch
from torchvision import transforms

import cred
import cfg


class LoadLocalDataset(Dataset):
    '''Load a dataset from local path. The given folder path is expected to contain a "train" and a "test" folder. Each of the folder will have subfolders (named according to their classes) and relevant images in them.
    '''

    def __init__(self, dataset_folder_path, image_height, image_width, image_depth, train=True, transform=None):
        '''Init parameters.
        '''

        assert not dataset_folder_path is None, "Path to the dataset folder must be provided!"

        self.dataset_folder_path = dataset_folder_path
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()
        self.train_transform = transforms.Compose([
                                             transforms.ToPILImage(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=cfg.COLOR_JITTER_BRIGHTNESS, hue=cfg.COLOR_JITTER_HUE),
                                             transforms.RandomAffine(degrees=cfg.RANDOM_AFFINE_ROTATION_RANGE, translate=cfg.RANDOM_AFFINE_TRANSLATE_RANGE, scale=cfg.RANDOM_AFFINE_SCALE_RANGE),
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
                                         ]) 

        self.test_transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
                                        ])


    def get_classnames(self):
        '''Returns the name of the classes by iterating through the train folder.
        '''
        class_names = os.listdir(f"{self.dataset_folder_path.rstrip('/')}/train/")
        return class_names 
    

    def read_folder(self):
        '''Reads the images and their corresponding label (folder name).
        '''

        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/train/"
        else:
            #bdd100k_weather dataset has a validation folder with the labels instead.
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/val/"

        for x in glob.glob(folder_path + '**', recursive=True):
            
            if not x.endswith('jpg'):
                continue
            
            class_idx = self.classes.index(x.split('/')[-2])
            image_path_label.append((x, int(class_idx)))

        return image_path_label

    def __len__(self):

        return len(self.image_path_label)

    def __getitem__(self, idx):
        '''Returns a single image array.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.image_path_label[idx]

        if self.image_depth == 1:
            image = cv2.imread(image, 0)
        
        else:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_height, self.image_width))

        if self.train:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
        
        #label = torch.tensor(float(label))

        return {
                'images': image,
                'labels': label
                }




class LoadDeeplakeDataset:
    '''Load a dataset from the Deeplake API.
    '''

    def __init__(self, token, deeplake_ds_name, batch_size, shuffle, mode='train'):
        '''Param init.
        '''

        self.token = token
        self.deeplake_ds_name = deeplake_ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode


    def collate_fn(self, batch_data):
        '''Custom collate function to preprocess the batch dataset.
        '''
        return {
                'images': torch.stack([x['images'] for x in batch_data]),
                'labels': torch.stack([torch.from_numpy(x['labels']) for x in batch_data])
            }

    @staticmethod
    def training_transformation():

        return transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=cfg.COLOR_JITTER_BRIGHTNESS, hue=cfg.COLOR_JITTER_HUE),
            transforms.RandomAffine(degrees=cfg.RANDOM_AFFINE_ROTATION_RANGE, translate=cfg.RANDOM_AFFINE_TRANSLATE_RANGE, scale=cfg.RANDOM_AFFINE_SCALE_RANGE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
        ])

    @staticmethod
    def testing_transformation():
        return  transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
        ])

    def __call__(self):

        deeplake_dataset = deeplake.load(self.deeplake_ds_name, token=self.token)

        dataloader = None
        if self.mode == 'train':
            dataloader = deeplake_dataset.dataloader().transform({'images':self.training_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})
        else:
            dataloader = deeplake_dataset.dataloader().transform({'images':self.testing_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})

        return dataloader



















