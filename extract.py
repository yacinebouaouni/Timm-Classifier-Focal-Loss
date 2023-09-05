import json
import random

import numpy as np
import torch
from dataset import DatasetBinary
from model import Classifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import argparse
from helpers import filter_df
from tqdm import tqdm
from torch import nn
def main(json_path='config/test.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--model', type=str, required=True, help="path to model")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    config = config['data']
    df = filter_df(config)
    df.to_csv('test_data.csv', index=None)

    m = Classifier("resnet18", 1).to(device)
    m.load_state_dict(torch.load(args.model))
    embedder = nn.Sequential(*list(m.children())[:-2])

    dataset_test = DatasetBinary(config, "test")
    L = len(dataset_test)

    m.eval()  # Set the model to evaluation mode

    embeddings = dict()
      
    for idx in tqdm(range(L)):

        output = dataset_test[idx]#{'img': img, 'label': label, 'path': img_path}
        img = output['img'].unsqueeze(dim=0).to(device)
        label = output['label']
        prediction = m(img).cpu().detach().numpy()[0][0]
        feature = embedder(img).cpu().detach().numpy().squeeze()[0]

        embeddings[output['path'].split('/')[-1][:-4]] = {
            "path":output['path'],
            "label":label,
            "prediction":float(prediction),
            "feature": feature.tolist()
        }

    with open("test_embeddings"+'.json', "w") as write_file:
        json.dump(embeddings, write_file)

if __name__ == '__main__':
    main()