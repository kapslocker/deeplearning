# use python3 division and print in python2
from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from prepare_data import *
import os
import copy
import argparse


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in xrange(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        ''' Train and test in every epoch. (To get stats after every epoch). '''
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                ''' Run over data (SGD) '''
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    ''' Perform backprop and update learning rate '''
                    loss.backward()
                    optimizer.step()
                ''' Evaluate loss '''
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('Phase: {}, Loss: {} Acc: {}'.format(
                phase, epoch_loss, epoch_acc))

            ''' Save the best model '''
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print('best test accuracy: {}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    ''' Fine tuning three networks (Transfer Learning) by adding a FC layer'''
    ''' Load dataset '''
    data_dir = 'data'
    splits = ['train', 'test']
    dataset = fetch_data(data_dir, splits)
    dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=4, shuffle=True, num_workers=4) for x in splits}
    dataset_sizes = {x: len(dataset[x]) for x in splits}
    num_classes = len(dataset['train'].classes)
    class_names = dataset['train'].classes

    use_gpu = torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type = str, help = "Which pretrained model to use: (\'resnet\', \"vgg16\", \'alexnet\')")
    args = parser.parse_args()
    if args.model == 'resnet':
        ''' Downloaded from: https://download.pytorch.org/models/resnet18-5c106cde.pth '''
        model_ft = models.resnet18(pretrained=False)
        model_ft.load_state_dict(torch.load('models/resnet18-5c106cde.pth'))
    elif args.model == 'vgg16':
        ''' Downloaded from: https://download.pytorch.org/models/vgg16-397923af.pth '''
        model_ft = models.vgg16(pretrained = False)
        model_ft.load_state_dict(torch.load('models/vgg16-397923af.pth'))
    elif args.model == 'alexnet':
        ''' Downloaded from https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth '''
        model_ft = models.alexnet(pretrained = False)
        model_ft.load_state_dict(torch.load('models/alexnet-owt-4df8aa71.pth'))

    else:
        print("Invalid choice of model")
        exit(0)

    ''' Add a fc layer to the model '''
    if args.model == 'vgg16':
        num_ftrs = model_ft.classifier[0].out_features
    else:
        num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.cuda()
    ''' Train to minimize the cross entropy loss '''
    criterion = torch.nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    ''' Exponential decay of learning rate by a factor of 0.1 every 5 epochs '''
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
    ''' Save best model to file '''
    torch.save(model_ft, 'models/saved_model_' + args.model + '.pth')
