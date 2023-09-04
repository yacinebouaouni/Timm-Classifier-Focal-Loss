from dataset import DatasetBinary
from model import Classifier
from loss.loss import select_loss
import torch.optim as optim
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config/train.json', 'r') as config_file:
    config = json.load(config_file)

config_loss = config['loss']
config = config['data']
m = Classifier("resnet50", 1).to(device)
d = DatasetBinary(config)
l = select_loss(config_loss).to(device)
optimizer = optim.Adam(m.parameters(), lr=0.001)

from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm.notebook for Jupyter compatibility

train_loader = DataLoader(d, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
validation_loader = DataLoader(d, batch_size=32, shuffle=True, num_workers=4)



# Number of training epochs
N = 10

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


    # Validation loop
    m.eval()  # Set the model to evaluation mode
    validation_loss = 0.0

    with torch.no_grad():
        for data in validation_loader:
            output = m(data['img'].to(device))
            loss = l(output.squeeze(), data['label'].to(device))
            validation_loss += loss.item()

    # Calculate and print average validation loss for the epoch (if validation data is available)
    if validation_loader:
        average_validation_loss = validation_loss / len(validation_loader)
        print(f"Epoch {epoch}/{N-1}, Train Loss : {average_loss:.4f}, Validation Loss: {average_validation_loss:.4f}")

# Training is complete
print("Training complete.")



