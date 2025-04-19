import os
from glob import glob

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
  def __init__(self, data_dir, train=True):
    file_names = glob(os.path.join(data_dir, '*', 'input', '*.*'))
    if train:
      self.img_names = [s for s in file_names if s[-8] == '0'] # include only training images
    else:
      self.img_names = [s for s in file_names if s[-8] != '0'] # include only test images
    self.label_names = [fname.replace(os.path.join('input','in'),os.path.join('groundtruth','gt')).replace('jpg','png') for fname in self.img_names]

    self.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((120,180))])
  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    img = self.transform(imageio.v3.imread(self.img_names[idx]))
    label = np.expand_dims(imageio.v3.imread(self.label_names[idx], mode='L'), axis=-1)

    # sparse indexing with many classes
    label[label == 50] = 1 # ignore
    label[label == 85] = 1 # ignore
    label[label == 170] = 1 # ignore
    label[label == 255] = 2
    label = torch.from_numpy(label).permute(2,0,1)
    label = transforms.Resize((120,180), interpolation=transforms.InterpolationMode.NEAREST)(label)

    return img, label