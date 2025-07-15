# Project: Vegetation Structure Estimation (Backup Algorithm)
# File:    predict_backup.py
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
from CNNmodel import BinaryCNN  
from glob import glob
import numpy as np
import pandas as pd

# Function to load and preprocess a single image
def load_pic(images_path):
    image = torch.tensor(np.load(images_path, allow_pickle=True).astype(np.float32))
    # lv, ls, dv, ds
    lv, dv = image[:,:,0], image[:,:,2]
    return lv + dv

def traversal_files(path):
        dirs, files = [], []
        for item in os.scandir(path):
            if item.is_dir():
                dirs.append(item.path)
            elif item.is_file():
                files.append(item.path)
        return dirs, files



if __name__ == "__main__":

    # Path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pic_dirs_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Data'))
    pth_path = os.path.abspath(os.path.join(base_dir, 'pth', 'backup_algorithm_cnn.pth'))

    dirs, files = traversal_files(pic_dirs_path)

    # Initialize the model
    model = BinaryCNN()
    weight = torch.load(pth_path)
    model.load_state_dict(weight)
    model.eval()
    lai, mta, lai_r, mta_r = [], [], [], []
    for i in range(len(dirs)):
        csv_dir = dirs[i] + ".csv"
        csv_data = pd.read_csv(csv_dir)
        LAI_R = float(csv_data['lai'][0])
        MTA_R = float(csv_data['mta'][0]) 

        """
        The first image was selected for testing, 
        for the alternate algorithm only the vegetation/soil component was calculated, 
        so all images with different lighting conditions were consistent
        """
        npy_path = glob(dirs[i] + '/*.npy')[0]
        try:
            contact_image = load_pic(npy_path)
        except:
            continue
        contact_image = contact_image.unsqueeze(0)
        with torch.no_grad():
            output = model(contact_image)
        LAI = float(output[0][0]) * 4
        MTA = float(output[0][1]) * 90
        lai.append(LAI)
        mta.append(MTA)
        lai_r.append(LAI_R)
        mta_r.append(MTA_R)

    print(f'Predicted LAI: {lai[0]:.2f} \n'
      f'True LAI: {lai_r[0]:.2f} \n'
      f'Predicted MTA: {mta[0]:.2f} \n'
      f'True MTA: {mta_r[0]:.2f}')


