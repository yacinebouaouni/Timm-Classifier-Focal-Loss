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
def main(json_path='config/test.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--model', type=str, required=True, help="name of experiment")
    args = parser.parse_args()

    # Set random seeds for PyTorch, random, and numpy
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set determinism for cudnn (GPU operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    config = config['data']
    df = filter_df(config)
    df.to_csv('test_data.csv', index=None)

    m = Classifier("resnet18", 1).to(device)
    m.load_state_dict(torch.load(args.model))

    dataset_test = DatasetBinary(config, "test")
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False,
                            drop_last=False, num_workers=8)


    m.eval()  # Set the model to evaluation mode
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for data in test_loader:
            output = m(data['img'].to(device))
            # Convert the model's output to binary predictions (e.g., 0 or 1)
            predictions.extend((output > 0.5).cpu().numpy().tolist())
            ground_truths.extend(data['label'].cpu().numpy().tolist())

        # Calculate accuracy and F1 score
        accuracy = accuracy_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions)
        print(f"Test f1 = {f1:.2f}")



    print("Testing complete.")




if __name__ == '__main__':
    main()