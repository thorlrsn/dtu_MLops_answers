import argparse
import sys
import torch
import click
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
sys.path.insert(1, r"C:\Users\Thor\Documents\dtu_MLops_answers\S1\final_exercise\model.py")
from model import MyAwesomeModel
import model as Mymodel
from torch import optim, nn
 
import time
import calendar

def train():
    print("Training...")

    # Given model
    # model = fc_model.Network(784, 10, [512, 256, 128])

    # Own model
    model = Mymodel.MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_set, test_set = mnist()
    
    # Given training
    # fc_model.train(model, train_set, test_set, criterion, optimizer, epochs=2)

    # Own training (The same as the given, since its pretty awesome)
    Mymodel.train(model, train_set, test_set, criterion, optimizer, 1)

    # Saving model
    save_model(model)

# custom functions
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Mymodel.MyAwesomeModel()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_model(model):
    # Giving values but they are not used.
    checkpoint = {'input_size': 1,
              'output_size': 10,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, "saved_model.pth")

def mnist():
    """ Using corrupted dataset """
    class MyDataset(Dataset):
        def __init__(self, path, train):
            datas = []
            if train:
                datas = []
                for i in range(5):
                    datas.append(np.load((path + str(i) + ".npz"), allow_pickle=True))
                self.imgs = torch.tensor(np.concatenate([c['images'] for c in datas])).reshape(-1, 1, 28, 28)
                self.labels = torch.tensor(np.concatenate([c['labels'] for c in datas]))
            else:
                data = np.load(path)
                self.imgs = data['images']
                self.imgs = torch.tensor(self.imgs).reshape(-1, 1, 28, 28)
                self.labels = data['labels']

        def __len__(self):
            return self.imgs.shape[0]

        def __getitem__(self, idx):
            return self.imgs[idx], self.labels[idx]

    train_path = r"C:\Users\Thor\Documents\dtu_MLops_answers\S1\final_exercise\data\data_corrupt\corruptmnist\train_"
    test_path = r"C:\Users\Thor\Documents\dtu_MLops_answers\S1\final_exercise\data\data_corrupt\corruptmnist\test.npz"

    train_data = MyDataset(train_path, train=True)

    test_data = MyDataset(test_path, train=False)
    
    trainloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

    """Using normal dataset """
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    # # Download and load the training data
    # trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=ToTensor())
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # # Download and load the test data
    # testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=ToTensor())
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader


if __name__ == "__main__":
    train()


    
    
    
    