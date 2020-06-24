import argparse
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, models, transforms
from utils import parse_gpu_arg, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('--data_dir', type=str, help='File path to dataset')
    parser.add_argument('--save_dir', type=str, help='Save trained model checkpoint', default="ImageClassifier/checkpoint.pth")
    parser.add_argument('--arch', type=str, help='Model Architecture', default='densenet121', choices=['densenet121', 'vgg13'])
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=512)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=9)
    parser.add_argument('--gpu', type=str, help='Make use of GPU if available (true|false), (yes|no), (y|n), (1|0)', default='true')

    return parser.parse_args()


def build_network(model, h_layers, model_choice, dropout=0.3):
    '''
    This function builds a Network with a pretrained model using default hidden layers
    and dropout percentage

    Paramaters
        h_layers:  Set number of hidden layers in the network
        dropout:   Set percentage of dropout per layer
    '''

    # Building Network
    # freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    if model_choice == 'densenet121':
        input_size = model.classifier.in_features

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, h_layers)),
            ('relu1', nn.ReLU()),
            ('dropout2', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(h_layers, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif model_choice == 'vgg13':
        input_size_vgg = model.classifier[0].in_features

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size_vgg, 1024)),
            ('drop', nn.Dropout(p=0.5)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(1024, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier
    #     model.class_to_idx = class_to_idx
    return model


# def get_model(model=models, model_type=args.arch):
#     '''
#     This function returns a default pretrained model (DenseNet121)

#     Paramaters
#         model:        Gets models from TorchVision modesl module
#         model_type:   Specifies the type of model to be used
#     '''
#     model = getattr(model, model_type)(pretrained=True)
# #     return model

def train_network(model, criterion, optimizer, epochs, trainloader, validloader, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU via cuda

    # Train Network using Back-Propagation and transfered learning from pre-trained network
    steps = 0
    running_loss = 0
    print_every = 40  # 5

    start_time = time.time()
    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Perform Forward-Pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Perform Forward-Pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)  # Use GPu if requested and available

                        logps = model.forward(inputs)
                        v_loss = criterion(logps, labels)
                        valid_loss += v_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print('Epoch {}/{}.. '.format(e + 1, epochs),
                      'Loss: {:.3f}.. '.format(running_loss / print_every),
                      'Validation Loss: {:.3f}.. '.format(valid_loss / len(validloader)),
                      'Accuracy: {:.3f}'.format(accuracy / len(validloader)))
                running_loss = 0
                model.train()

    total_train_time = time.time() - start_time
    print(
        '\n\nTotal time taken to train network: {:.0f}m {:.2f}s'.format(total_train_time // 60, total_train_time % 60))


def main():
    print('Welcome')

    args = parse_args()

    #     data_dir = 'flowers'
    rgs = parse_args()
    if args.data_dir:
         data_dir = args.data_dir
    else:
        raise argparse.ArgumentError(args.data_dir, 'Path to training dataset required')
   
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets, load image into dataloader
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=64)

    model = getattr(models, args.arch)(pretrained=True)    # Get model
    
    if args.arch == 'densenet121':
        model_choice = 'densenet121'
    else:
        model_choice =  'vgg13'
        
    model = build_network(model, args.hidden_units, model_choice)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    gpu = parse_gpu_arg(args.gpu)  # Make use of GPU if available

    train_network(model, criterion, optimizer, epochs, trainloader, validloader, gpu)
    model.class_to_idx = train_data.class_to_idx
    save_path = args.save_dir

    save_checkpoint(save_path, model, optimizer, model.classifier, args)


if __name__ == "__main__":  # Only if this module is in main program, will call main() function
    main()
