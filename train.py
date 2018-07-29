import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import json
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, help='NN model name')
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--epochs', type=int, help='epochs for training')
parser.add_argument('--data_dir', type=str, help='dataset dir')
parser.add_argument('--gpu', action='store_true', help='GPU computation')
parser.add_argument('--checkpoint', type=str, help='checkpoint')
parser.add_argument('--hidden_units', type=int, help='count of hidden units')
args, _ = parser.parse_known_args()

def load_model(arch='vgg19', num_labels=102, hidden_units=4096):

    if arch=='alexnet':
        model = models.alexnet(pretrained=True)
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError('unknown model')
    
    for param in model.parameters():
        param.requires_grad = False
    
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Dropout(), nn.Linear(model.classifier[len(features)].in_features, hidden_units), nn.ReLU(True), nn.Dropout(), nn.Linear(hidden_units, hidden_units), nn.ReLU(True), nn.Linear(hidden_units, num_labels)])
    model.classifier = nn.Sequential(*features)

    return model



def train_model(arch='vgg19', learning_rate=0.001, epochs=25, image_datasets, gpu=False, checkpoint='', hidden_units=4096):
    if args.arch:
        arch = args.arch     
    if args.learning_rate:
        learning_rate = args.learning_rate    
    if args.epochs:
        epochs = args.epochs
    if args.gpu:
        gpu = args.gpu
    if args.checkpoint:
        checkpoint = args.checkpoint        
    if args.hidden_units:
        hidden_units = args.hidden_units
        

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    test_dataloaders  = torch.utils.data.DataLoader(image_datasets['test'],  batch_size=64, shuffle=True)

    dataloaders = {
        'train' : train_dataloaders,
        'valid' : valid_dataloaders,
        'test'  : test_dataloaders
    }

    dataset_sizes = {
        'train' : len(dataloaders['train'].dataset),
        'valid' : len(dataloaders['valid'].dataset),
        'test'  : len(dataloaders['test'].dataset) 
    }

    model = load_model(arch=arch, num_labels=len(image_datasets['train'].classes), hidden_units=hidden_units)
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.cuda()
    else:
        device = torch.device("cpu")     

    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)    
    criterion = nn.CrossEntropyLoss()
        
    wts_best_model = copy.deepcopy(model.state_dict())
    highest_acc = 0.0
    print('Training the model')
    for epoch in range(epochs):
        print('epoch {} out of {}'.format(epoch + 1, epochs))

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:                
                labels = labels.to(device)
                inputs = inputs.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} loss: {:.4f} accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > highest_acc:
                highest_acc = epoch_acc
                wts_best_model = copy.deepcopy(model.state_dict())

    print('best value accuracy: {:4f}'.format(highest_acc))

    model.load_state_dict(wts_best_model)
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    if checkpoint:
        print ('save checkpoint ', checkpoint) 
        checkpoint_dict = {
            'arch': arch,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units
        }
        
        torch.save(checkpoint_dict, checkpoint)

    return model






if args.data_dir:

    train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms = {
        'train' : train_transforms,
        'valid' : valid_transforms,
        'test'  : test_transforms
    }
    
    image_datasets = {
        i: datasets.ImageFolder(root=args.data_dir + '/' + i, transform=data_transforms[i])
        for i in list(data_transforms.keys())
    }
        
    train_model(image_datasets)