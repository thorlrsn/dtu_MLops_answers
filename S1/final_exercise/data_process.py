import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, path, train):
        datas = []
        if train:
            datas = []
            for i in range(5):
                datas.append(np.load((path + str(i) + ".npz"), allow_pickle=True))
                print(path + str(i) + ".npz")
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
# print(type(train_data))
first_data = train_data[0]
features, labels = first_data
# print(features, labels)
trainloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
print(type(trainloader))
