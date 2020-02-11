import os

import numpy as np

import torch
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models
from models import Net, EMNISTGoogLeNet

# Performance metrics
import utils

# For progress bar
from tqdm.autonotebook import tqdm


def preprocess_and_load_train_data(data_path, batch_size, shuffle=True, use_val=False, val_split=0.0):
    # Convert images to tensors, after applying random transformations, and normalise
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1745,), (0.3223,))
    ])

    # Using updated link for dataset: 
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download

    dataset = datasets.EMNIST(root=data_path, split='byclass', train=True,
                                download=True, transform=transform)

    if use_val:
        # Split data into train and val sets
        dataset_len = len(dataset)
        val_len = int(np.floor(val_split * dataset_len))
        train_len = dataset_len - val_len
        train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                    shuffle=shuffle, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                    shuffle=shuffle, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    shuffle=shuffle, num_workers=2)
        val_loader = None

    return train_loader, val_loader


def preprocess_and_load_test_data(data_path, batch_size, shuffle=True):
    # Convert images to tensors and normalise
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1745,), (0.3223,))
    ])

    # Using updated link for dataset: 
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download

    dataset = datasets.EMNIST(root=data_path, split='byclass', train=False,
                                download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=2)

    return dataloader
            

def train_model(net, criterion, optimizer, epochs, data_loaders, device, model_path):
    min_val_loss = np.Inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch + 1}/{epochs}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
            else:
                if val_loader is None:
                    # Skip val step
                    continue
                net.train(False)
            
            batches = len(data_loaders[phase]) 
            stage_loss = "Train Loss" if phase == 'train' else "Val Loss"
            progress = tqdm(enumerate(data_loaders[phase]), desc=f"{stage_loss}: ", total=batches)

            running_loss = 0.0
            for i, data in progress:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                # update progress bar
                progress.set_description("{}: {:.4f}".format(stage_loss, running_loss/(i + 1)))

            if phase == 'val':
                val_loss = running_loss / len(data_loaders['val'])
                print("Val loss at end of epoch: ", val_loss)
                if val_loss < min_val_loss:
                    print(f"Validation loss improved from {min_val_loss:.4f} to {val_loss:.4f}")
                    print("Saving best model...")
                    utils.save_model(net, optimizer, model_path)
                    epochs_no_improve = 0
                    min_val_loss = val_loss
                else:
                    print(f"Validation loss {val_loss:.4f} did not improve from {min_val_loss:.4f}")
                    epochs_no_improve += 1
                
                if epochs_no_improve == data_loaders['patience']:
                    print("Early stopping!")
                    early_stop = True
                    break

        if early_stop:
            print("Exiting main training loop")
            break

    if not early_stop:
        utils.save_model(net, optimizer, model_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Finished Training')


if __name__ == "__main__":
    # Pytorch configuration
    random_seed = 1
    torch.manual_seed(random_seed) # For model reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    hyperparameters = {
        'epochs' : 150,
        'batch_size_train' : 64,
        'batch_size_test' : 1000,
        'learning_rate' : 0.01
    }

    # Where to save EMNIST train and test data
    DATA_BASE_PATH = os.getenv('PROJECT_DATA_BASE_DIR')
    data_path = os.path.abspath(os.path.join(DATA_BASE_PATH, 'EMNIST'))

    print("Loading and preprocessing data...")
    train_loader, val_loader = preprocess_and_load_train_data(
        data_path, hyperparameters['batch_size_train'], 
        use_val=True, val_split=0.15
    )

    net = EMNISTGoogLeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hyperparameters['learning_rate'])

    print("Training model...")
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    model_name = 'emnist_centralised_inception.pth'
    model_path = os.path.abspath(os.path.join(curr_file_path, '..' , f'saved_models/{model_name}'))

    data_loaders = {'train': train_loader, 'val': val_loader, 'patience': 10}
    train_model(
        net, criterion, optimizer, hyperparameters['epochs'], 
        data_loaders, device, model_path
    )

    print("Evaluating model...")
    test_loader = preprocess_and_load_test_data(data_path, hyperparameters['batch_size_test'], shuffle=False)
    net = EMNISTGoogLeNet().to(device)
    utils.load_model(net, model_path)
    utils.evaluate_model(net, test_loader, device)
