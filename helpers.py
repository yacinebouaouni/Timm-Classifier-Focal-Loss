import numpy as np
import torch
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def filter_df(config):

    df = pd.read_csv(config['dataroot']['csv'])
    df = df.loc[df.lib == config['DIE']]
    df = df.loc[df.program == config['program']]
    dfs = []
    for year in list(config['scope'].keys()):
        for month in config['scope'][year]:
            dfs.append(df.loc[(df.year == int(year)) & (df.month == int(month))])
    dfs = pd.concat(dfs)
    return dfs.reset_index()

def split_data(config):
    data = filter_df(config)

    # Group the data by class labels
    grouped_data = data.groupby('Sanction')

    # Initialize empty DataFrames for training and evaluation sets
    train_data = pd.DataFrame()
    eval_data = pd.DataFrame()

    # Split each class while maintaining class balance
    for _, group in grouped_data:
        # Split the class into training and evaluation sets (adjust test_size as needed)
        train_class, eval_class = train_test_split(group, test_size=0.1, random_state=42, stratify=group['Sanction'])

        # Append the split class data to the training and evaluation DataFrames
        train_data = pd.concat([train_data, train_class], ignore_index=True)
        eval_data = pd.concat([eval_data, eval_class], ignore_index=True)

            # Reset the index for the training and evaluation DataFrames and drop the original index

    train_data.drop(columns=["index"], inplace=True)
    eval_data.drop(columns=["index"], inplace=True)
    # Save the training and evaluation sets to separate CSV files
    train_data.to_csv('train_data.csv', index=None)
    eval_data.to_csv('eval_data.csv', index=None)
