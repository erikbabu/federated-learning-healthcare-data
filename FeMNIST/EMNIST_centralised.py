import os

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models.googlenet import GoogLeNet

# Performance metrics
from utils import overall_model_performance, class_based_model_performance, CLASSES

# For progress bar
from tqdm.autonotebook import tqdm

# TODO: Move to separate file
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 62)
        self.pool = nn.MaxPool2d(2, 2)
#         self.drop_out = nn.Dropout()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class EMNISTGoogLeNet(GoogLeNet):
    def __init__(self):
        super(EMNISTGoogLeNet, self).__init__(num_classes=len(CLASSES), aux_logits=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        return F.log_softmax(
            super(EMNISTGoogLeNet, self).forward(x), dim=1
        )


def preprocess_and_load_train_data(data_path, batch_size, shuffle=True, use_val=False, val_split=0.0):
    # Convert images to tensors and normalise (implicitly) in range [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])

    # Using updated link for dataset: 
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download

    dataset = torchvision.datasets.EMNIST(root=data_path, split='byclass', train=True,
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
    # Convert images to tensors and normalise (implicitly) in range [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])

    # Using updated link for dataset: 
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download

    dataset = torchvision.datasets.EMNIST(root=data_path, split='byclass', train=False,
                                        download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=2)

    return dataloader
            

def train_model(net, criterion, optimizer, epochs, data_loaders, device):
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

            net.train()
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


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Finished Training')


def save_model(net, optimizer, model_path):
    torch.save({ 
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)


def load_model(net, path):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])


def evaluate_model(net, test_loader, device):
    net.eval()
    overall_model_performance(net, test_loader, device)
    class_based_model_performance(net, test_loader, device)
    


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
    data_loaders = {'train': train_loader, 'val': val_loader, 'patience': 10}
    train_model(
        net, criterion, optimizer, hyperparameters['epochs'], 
        data_loaders, device
    )

    print("Saving model...")
    model_path = 'emnist_centralised.pth'
    save_model(net, optimizer, model_path)

    print("Evaluating model...")
    test_loader = preprocess_and_load_test_data(data_path, hyperparameters['batch_size_test'], shuffle=False)
    net = EMNISTGoogLeNet().to(device)
    load_model(net, model_path)
    evaluate_model(net, test_loader, device)
