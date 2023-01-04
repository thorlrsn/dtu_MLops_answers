import argparse
import sys
import helper
import torch
import click
import fc_model
from data import mnist
from model import MyAwesomeModel
import model as Mymodel
from torch import optim, nn

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
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
    save_checkpoint(model)

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating...")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    _ , test_set = mnist()
    criterion = nn.CrossEntropyLoss()
    model = load_checkpoint('checkpoint.pth')
    model.eval()
                
    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = Mymodel.validation(model, test_set, criterion)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_set)),
            "Test Accuracy: {:.3f}".format(accuracy/len(test_set)))

cli.add_command(train)
cli.add_command(evaluate)

# custom functions
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Mymodel.MyAwesomeModel()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_checkpoint(model):
    # Giving values but they are not used.
    checkpoint = {'input_size': 1,
              'output_size': 10,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    cli()


    
    
    
    