import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
def mnist():
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784) 

    # Didnt work very well, couldnt create datasest in such a way that batch training could be used.
    # train_path = r"C:\Users\thorl\Documents\DTU\JAN23\dtu_mlops\data\corruptmnist\train_0.npz"
    # test_path = r"C:\Users\thorl\Documents\DTU\JAN23\dtu_mlops\data\corruptmnist\test.npz"

    # train = np.load(train_path, allow_pickle=True)
    # test = np.load(test_path, allow_pickle=True)

    ### Using normal dataset ###

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader
