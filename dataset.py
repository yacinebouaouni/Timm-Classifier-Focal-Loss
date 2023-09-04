import torch.utils.data as data
import cv2
import pandas as pd
import os
from sklearn import preprocessing
from helpers import uint2tensor3, imread_uint




class DatasetBinary(data.Dataset):
    '''
    # -----------------------------------------
      |Root 
      |____ YYYY
      |         |____ MM
      |                 |____ DD
    # -----------------------------------------
    This dataset takes a root and combines multiple years/months/days
    Train: YYYY MM DD
    '''

    def __init__(self, config, tag):
        super(DatasetBinary, self).__init__()
        self.config = config
        self.n_channels = self.config['n_channels'] if self.config['n_channels'] else 1
        self.img_size = self.config['size'] if self.config['size'] else 512
        self.path_csv = self.config['dataroot'][tag]
        self.root = self.config['dataroot']['root']
        # ------------------------------------
        # get the path of the images
        # ------------------------------------
        df = pd.read_csv(self.path_csv)
        self.paths = list(df['path'].values)
        self.labels = list(df['Sanction'].values)

        if self.config['encoding']:
            print('Encoding Labels Using Provided Dict')
            self.labels = [self.config['encoding'][k] for k in self.labels]            
        else:
            print('Encoding Labels Using LabelEncoder')
            encoder = preprocessing.LabelEncoder()
            self.labels = encoder.fit_transform(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root,self.paths[index])
        return {'img': self.processing(img_path), 'label': self.labels[index], 'path': img_path}

    def __len__(self):
        return len(self.paths)

    def processing(self, path):
        img = imread_uint(path, self.n_channels) #numpy [0-255] HxWxC
        img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_AREA)
        img = uint2tensor3(img) #tensor CxHxW 0-1
        return img

  
