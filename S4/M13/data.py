import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

def mnist(bs=64):
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

    train_path = r"C:\Users\thorl\Documents\DTU\JAN23\dtu_MLops_answers\S1\final_exercise\data\data_corrupt\corruptmnist\train_"
    test_path = r"C:\Users\thorl\Documents\DTU\JAN23\dtu_MLops_answers\S1\final_exercise\data\data_corrupt\corruptmnist\test.npz"
    train_data = MyDataset(train_path, train=True)

    test_data = MyDataset(test_path, train=False)
    
    trainloader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=bs, shuffle=True)

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
