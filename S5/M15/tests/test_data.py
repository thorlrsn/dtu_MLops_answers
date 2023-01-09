import os
import pytest
import sys
from torch.utils.data import Dataset, DataLoader
import torch

# Paths are messed up so had to add manually, not ideal but works for now
sys.path.insert(0,r"C:\Users\Thor\Documents\dtu_MLops_answers\S5\M15")
###
# The paths work correctly when running "pytest .\tests\" from the terminal,
# it is an internal issue when running individual files
from data import mnist
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_length():
    print(_PATH_DATA)
    N_train = 25000
    N_test = 5000

    # mnist function is modified to return raw data and not DataLoader as it normally does
    data_train, data_test = mnist(_PATH_DATA)

    # trainloader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    # testloader = DataLoader(dataset=test_data, batch_size=bs, shuffle=True)

    assert len(data_train) == N_train, "Dataset did not have the correct number of samples"

    assert len(data_test) == N_test, "Dataset did not have the correct number of samples"

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_shape():
    data_train, _ = mnist(_PATH_DATA)
    
    #batch size
    bs = 64
    
    trainloader = DataLoader(dataset=data_train, batch_size=bs, shuffle=True)
    # testloader = DataLoader(dataset=data_test, batch_size=bs, shuffle=True)
    for images, labels in trainloader:
        images = images.float()
        labels = labels.long()

        #image shape
        img_shape = list(images.shape)

        #labels shape
        lbs_shape = list(labels.shape)

        assert img_shape == [bs, 1, 28, 28], "Dataset did not have the correct shape of dimensions"
        assert lbs_shape == [bs],  "Dataset did not have the correct shape of dimensions"

        break #so we dont end in a loop, we only check for one image and label

