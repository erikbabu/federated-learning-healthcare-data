import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import overall_model_performance, class_based_model_performance

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


def preprocess_and_load_data(data_path, batch_size, train=True, shuffle=True):
    # Convert images to tensors and normalise (implicitly) in range [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])

    # Using updated link for dataset: 
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download

    dataset = torchvision.datasets.EMNIST(root=data_path, split='byclass', train=train,
                                        download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=2)

    return dataloader
            

def train_model(net, criterion, optimizer, epochs, trainloader, device):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def save_model(net, optimizer, model_path):
    torch.save({ 
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)


def load_model(net, path):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])


def evaluate_model(net, testloader, device):
    net.eval()
    overall_model_performance(net, testloader, device)
    class_based_model_performance(net, testloader, device)
    


if __name__ == "__main__":
    # Pytorch configuration
    random_seed = 1
    torch.manual_seed(random_seed) # For model reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    hyperparameters = {
        'epochs' : 3,
        'batch_size_train' : 64,
        'batch_size_test' : 1000,
        'learning_rate' : 0.01
    }

    # Where to save EMNIST train and test data
    DATA_BASE_PATH = os.getenv('PROJECT_DATA_BASE_DIR')
    data_path = os.path.abspath(os.path.join(DATA_BASE_PATH, 'EMNIST'))

    print("Loading and preprocessing data...")
    train_loader = preprocess_and_load_data(data_path, hyperparameters['batch_size_train'])

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hyperparameters['learning_rate'])

    print("Training model...")
    train_model(net, criterion, optimizer, hyperparameters['epochs'], train_loader, device)

    print("Saving model...")
    model_path = 'emnist_centralised.pth'
    save_model(net, optimizer, model_path)

    print("Evaluating model...")
    test_loader = preprocess_and_load_data(data_path, hyperparameters['batch_size_test'], train=False, shuffle=False)
    net = Net().to(device)
    load_model(net, model_path)
    evaluate_model(net, test_loader, device)
