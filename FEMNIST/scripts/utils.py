import torch

CLASSES = [str(i) for i in range(10)]
CLASSES += list(map(chr, range(65, 91)))
CLASSES += list(map(chr, range(97, 123)))


def dataset_statistics(loader):
    "Computes and returns mean and standard deviation of dataset"
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    return mean, std

def save_model(net, optimizer, model_path):
    if not model_path:
        return

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


def overall_model_performance(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def class_based_model_performance(net, testloader, device):
    class_correct = list(0. for i in range(len(CLASSES)))
    class_total = list(0. for i in range(len(CLASSES)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(predicted)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(len(CLASSES)):
        print('Accuracy of %5s : %2d %%' % (
            CLASSES[i], 100 * class_correct[i] / class_total[i]))