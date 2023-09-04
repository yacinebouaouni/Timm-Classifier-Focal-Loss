import json
import random

import numpy as np
import torch
import torch.optim as optim
from dataset import DatasetBinary
from loss.loss import select_loss
from model import Classifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from helpers import split_data
import os

def main(json_path='config/train.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--experiment', type=str, required=True, help="name of experiment")
    args = parser.parse_args()

    if not os.path.exists("logs"):
        os.makedirs(os.path.join('logs', args.experiment, "models"))
    elif args.experiment in os.listdir('logs'):
        raise FileExistsError(f"The folder '{args.experiment}' already exists. Please specify a different folder name.")
    else:
        os.makedirs(os.path.join('logs', args.experiment, "models"))
    

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

    with open('config/train.json', 'r') as config_file:
        config = json.load(config_file)

    config_loss = config['loss']
    config = config['data']

    split_data(config)
    # Create a TensorBoard writer
    writer = SummaryWriter(log_dir="./runs/"+args.experiment)
    m = Classifier("resnet50", 1).to(device)
    dataset_train = DatasetBinary(config, "train")
    dataset_eval = DatasetBinary(config, "validation")
    l = select_loss(config_loss).to(device)
    optimizer = optim.Adam(m.parameters(), lr=0.0001)


    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True,
                            drop_last=True, num_workers=8)
    validation_loader = DataLoader(dataset_eval, batch_size=64, shuffle=True, num_workers=8)

    # Number of training epochs
    N = 3

    for epoch in range(N):

        pb = tqdm(train_loader, unit="batch", ncols=100)

        pb.set_description(f"Epoch {epoch}")

        # Training loop
        m.train()  # Set the model to training mode
        total_loss = 0.0

        for data in pb:
            # Forward pass
            optimizer.zero_grad()
            output = m(data['img'].to(device))
            loss = l(output.squeeze(), data['label'].to(device))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pb.set_postfix(loss=f"{total_loss:.4f}")

        # Calculate and print average training loss for the epoch
        average_loss = total_loss / len(train_loader)
        # Log training loss to TensorBoard
        writer.add_scalar("Loss/Training", average_loss, epoch)

        # Validation loop
        m.eval()  # Set the model to evaluation mode
        validation_loss = 0.0
        predictions = []
        ground_truths = []
        with torch.no_grad():
            for data in validation_loader:
                output = m(data['img'].to(device))
                loss = l(output.squeeze(), data['label'].to(device))
                validation_loss += loss.item()
                # Convert the model's output to binary predictions (e.g., 0 or 1)
                predictions.extend((output > 0.5).cpu().numpy().tolist())
                ground_truths.extend(data['label'].cpu().numpy().tolist())

        # Calculate and print average validation loss for the epoch (if validation data is available)
        if validation_loader:
            average_validation_loss = validation_loss / len(validation_loader)
            # Calculate accuracy and F1 score
            accuracy = accuracy_score(ground_truths, predictions)
            f1 = f1_score(ground_truths, predictions)
            # Log validation loss, accuracy, and F1 score to TensorBoard
            writer.add_scalar("Loss/Validation", average_validation_loss, epoch)
            writer.add_scalar("Accuracy", accuracy, epoch)
            writer.add_scalar("F1 Score", f1, epoch)
            print(f"Epoch {epoch}/{N-1}, Train Loss : {average_loss:.4f}, Validation Loss: {average_validation_loss:.4f}, Validation f1 = {f1:.2f}")

            torch.save(m.state_dict(), os.path.join("logs", args.experiment, "models", "model_"+str(epoch)+".pth"))


    # Training is complete
    writer.close()
    print("Training complete.")




if __name__ == '__main__':
    main()