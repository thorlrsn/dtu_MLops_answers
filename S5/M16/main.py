import torch
import wandb
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from data import mnist
import model as Mymodel

from tests import _PATH_DATA

def train():
    # Given model
    # model = fc_model.Network(784, 10, [512, 256, 128])

    # Own model
    model = Mymodel.MyAwesomeModel()

    # Wandb stuff
    wandb.init()
    wandb.watch(model, log_freq=100)
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr  =  wandb.config.lr

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set, test_set = mnist(_PATH_DATA)
    trainloader = DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
    testloader = DataLoader(dataset=test_set, batch_size=bs, shuffle=True)
    
    # Given training
    # fc_model.train(model, train_set, test_set, criterion, optimizer, epochs=2)

    # Own training (The same as the given, since its pretty awesome)
    print("Training...")
    Mymodel.train(model, trainloader, testloader, criterion, optimizer, epochs)

    # Exporting table to wandb with sampled images, preditions and truths
    wandb_table(model, testloader)

    # Saving model
    # save_checkpoint(model)

def evaluate(model_checkpoint):
    print("Evaluating...")
    print(model_checkpoint)

    bs = wandb.config.batch_size
    _ , test_set = mnist(_PATH_DATA)
    testloader = DataLoader(dataset=test_set, batch_size=bs, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = model_checkpoint
    model.eval()
                
    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = Mymodel.validation(model, testloader, criterion)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

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

def wandb_table(model, testloader):
    example_images = []
    example_pred = []
    example_truth = []

    for images, labels in testloader:
        images = images.float()
        labels = labels.long()
        
        output = model.forward(images)

        ps = torch.exp(output)
        pred = ps.max(1)[1]

        # would like to perform the addition to the table here but it wont work.
        example_pred.append(pred[0].item())
        example_truth.append(labels[0].item())
        example_images.append(wandb.Image(images[0], caption="Pred: {} Truth: {}".format(pred[0].item(), labels[0])))
                
    columns = ["Image", "Prediction", "Truth"]
    table = wandb.Table(columns=columns)

    # Adding to table here instead in a small loop
    for i in range(len(example_pred)):
        table.add_data(example_images[i], example_pred[i], example_truth[i])
    
    wandb.log({"Preditions": table})

def wandb_config(model):
    # Old wandb config
    # args = {"batch_size": 64,  # try log-spaced values from 1 to 50,000
    #       "num_workers": 2,  # try 0, 1, and 2
    #       "pin_memory": False,  # try False and True
    #       "precision": 32,  # try 16 and 32
    #       "optimizer": "Adadelta",  # try optim.Adadelta and optim.SGD
    #       }
    # wandb.init()
    # wandb.init(config=args)
    # wandb.watch(model, log_freq=100)

    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [1, 3, 5]},
            'lr': {'max': 0.001, 'min': 0.0001}
        }
    }

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration)
    # wandb.init()
    return sweep_id


if __name__ == "__main__":
    # model = load_checkpoint(r"C:\Users\thorl\Documents\DTU\JAN23\dtu_MLops_answers\S4\M13\checkpoint.pth")
    model = Mymodel.MyAwesomeModel()
    sweep_id = wandb_config(model)
    wandb.agent(sweep_id, function=train, count=4)
    # train()

    
    
    
    