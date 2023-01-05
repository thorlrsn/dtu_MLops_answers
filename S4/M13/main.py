import torch
import wandb
from torch import optim, nn

from data import mnist
import model as Mymodel

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
    Mymodel.train(model, train_set, test_set, criterion, optimizer, 3)

    # exporting table to wandb with sampled images, preditions and truths
    wandb_table(model, test_set)
    # Saving model
    save_checkpoint(model)

def evaluate(model_checkpoint):
    print("Evaluating...")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    _ , test_set = mnist()
    criterion = nn.CrossEntropyLoss()
    model = model_checkpoint
    model.eval()
                
    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = Mymodel.validation(model, test_set, criterion)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_set)),
            "Test Accuracy: {:.3f}".format(accuracy/len(test_set)))

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

if __name__ == "__main__":
    model = load_checkpoint("S4/M13/checkpoint.pth")
    train()

    
    
    
    