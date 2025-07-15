# Project: Vegetation Structure Estimation (Backup Algorithm)
# File:    predict_main.py
# Author:  Feng Yaowei
# Date:    July 15, 2025
#
# Description:
#   This script predicts Leaf Area Index (LAI) and Mean Tilt Angle (MTA)
#   from binary vegetation component images using a CNN-based backup algorithm.
#   It is part of a larger framework for time-series vegetation structure analysis.
#

import os
import torch
from gru_model import GRUNet
import random
from glob import glob
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

def load_pics(pic_dir, pic_size, csv_load = False):
    images_path = glob(pic_dir + "/*_four.npy")
    contact_image = []
    percent_four = np.zeros((len(images_path), 6),dtype='float32')
    for i in range(len(images_path)):
        # Load image
        image = torch.tensor(np.load(images_path[i], allow_pickle=True).astype(np.uint8))
        # Split tensor into sub-tensors by channels (dim=2)
        sub_tensors = torch.split(image, split_size_or_sections=1, dim=2)
        # Compute the sum of each sub-tensor (representing one component)
        sums = [sub_tensor.squeeze().sum() for sub_tensor in sub_tensors]
        for j in range(len(sums)):
            percent_four[i][j] = sums[j]/(1024 * 1024)
        percent_four[i][4] = float(images_path[i].split('_')[-2])/180 # Azimuth angle
        percent_four[i][5] = float(images_path[i].split('_')[-3])/50 # Zenith angle
        image = image.permute(2, 0, 1)
        n_image = image.numpy()
        # Downscale image to match model input resolution
        output_array = zoom(n_image, [1,0.25,0.25], order=0)
        image = torch.from_numpy(output_array)
        contact_image.append(image)
    contact_image = torch.stack(contact_image, dim=0)
    csv_dir = pic_dir + ".csv"
    csv_data = pd.read_csv(csv_dir)
    LAI = float(csv_data['lai'][0])
    MTA = float(csv_data['mta'][0]) 
    return contact_image, percent_four, LAI, MTA

def traversal_files(path):
        dirs, files = [], []
        for item in os.scandir(path):
            if item.is_dir():
                dirs.append(item.path)
            elif item.is_file():
                files.append(item.path)
        return dirs, files


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pic_dirs_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Data'))
    pth_path = os.path.abspath(os.path.join(base_dir, 'pth', 'main_algorithm_gru.pth'))

    # # Path of model weights
    # pth_path = 'I:/transform/LSTM/Program/main_algorithm/pth/main_algorithm_gru.pth'

    # # Path of test data
    # pic_dirs_path = 'I:/transform/LSTM/Data/data1'

    dirs, files = traversal_files(pic_dirs_path)
    random.shuffle(dirs)
    model = GRUNet()
    weight = torch.load(pth_path)
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    lai, mta, lai_r, mta_r = [], [], [], []

    for i in range(len(dirs)):

        pic_dir_path = dirs[i]
        print(pic_dir_path)
        contact_image, percent_four, LAI_t, MTA_t = load_pics(pic_dir_path, pic_size = 512, csv_load = True)
        contact_image = contact_image.unsqueeze(0)
        percent_four = torch.tensor(percent_four)
        contact_image = contact_image.to(device)
        percent_four = percent_four.to(device)
        with torch.no_grad():
            output = model(contact_image, percent_four)
        output = output.cpu()
        # Calculate LAI and MTA
        LAI = float(output[0][0]) * 4
        MTA = float(output[0][1]) * 90
        
        lai.append(LAI)
        mta.append(MTA)
        lai_r.append(LAI_t)
        mta_r.append(MTA_t)


    
    print(f'Predicted LAI: {lai[0]:.2f} \n'
      f'True LAI: {lai_r[0]:.2f} \n'
      f'Predicted MTA: {mta[0]:.2f} \n'
      f'True MTA: {mta_r[0]:.2f}')
