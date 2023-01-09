from torch.utils.data import Dataset, DataLoader
from data import mnist
from tests import _PATH_DATA
import model as Mymodel
import pytest
import torch

def test_model_out():
    model = Mymodel.MyAwesomeModel()
    assert model.fc2.out_features == 10, "Output features wasnt the right size"

def test_error_on_wrong_shape():
    model = Mymodel.MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3,2,1))
        