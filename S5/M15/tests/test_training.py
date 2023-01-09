from torch.utils.data import Dataset, DataLoader
import torch
from data import mnist
from tests import _PATH_DATA
import os
import pytest
import model as Mymodel
import wandb
from torch import optim, nn
from main import wandb_config
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_training():
    bs = 64 
    train_set, test_set = mnist(_PATH_DATA)
    trainloader = DataLoader(dataset=train_set, batch_size=bs, shuffle=True)

    for images, labels in trainloader:
        # lets check if the data is given in tensor form      
        assert torch.is_tensor(images), "Data was not given in tensor form"
        assert torch.is_tensor(labels), "Data was not given in tensor form"

def test_one_epoch():
     # Own model
    model = Mymodel.MyAwesomeModel()
    
    # Wandb stuff
    wandb.init()
    wandb.watch(model, log_freq=100)
    bs = 64
    epochs = 1
    lr  =  0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set, test_set = mnist(_PATH_DATA)
    trainloader = DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
    testloader = DataLoader(dataset=test_set, batch_size=bs, shuffle=True)
    
    Mymodel.train(model, trainloader, testloader, criterion, optimizer, epochs)

    test_loss, test_acc = Mymodel.validation(model, testloader, criterion)

    assert test_acc > 0.8, "The test acc was too low to pass"

