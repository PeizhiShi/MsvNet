
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path
import numpy as np
from libs.utils import progress_bar
import random
import copy

from torchvision import transforms, utils
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
from collections import OrderedDict
import libs.dataset as lds
import libs.models as lmd
import warnings



def train(epoch, net, device, trainloader, criterion, optimizer, stage = 2):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        if stage == 1:
            inputs = inputs.view(-1,3,64,64)
            targets = targets.view(-1)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
      
        
        loss = criterion(outputs, targets)
        
        
        loss.backward()
            
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
        
def valtest(epoch, net, device, dataloader, criterion, testval = 'Val'):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    
    with torch.no_grad():
        for batch_idx, (inputs, targets_org) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets_org.to(device)
            
            
            outputs = net(inputs)

            
            
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            val, predicted = outputs.max(1)
            #print(val)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | %s Acc: %.3f%% (%d/%d)'
                % (val_loss/(batch_idx+1), testval, 100.*correct/total, correct, total)) 

    return 100.*correct/total
            

      

def train_test_mcnn(partition, parameters):
    
    learning_rate = parameters.learning_rate 
    epoch1  = parameters.epoch1
    epoch2  = parameters.epoch2
    resolution = parameters.resolution 
    num_cuts = parameters.num_cuts
    batch_size = parameters.batch_size
    
    
    training_set = lds.FeatureDataset(partition['train'], resolution, True, output_type = '2d_single', num_cuts = num_cuts, data_augmentation = True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size = batch_size, num_workers=2)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = lmd.SCNN(pretraining = parameters.pretrained)
    net.to(device)
    
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    
    if parameters.finetuned:
        print('Stage 1: fine tuning the neural network ...')
        for epoch in range(epoch1):
            train(epoch, net, device, trainloader, criterion, optimizer, stage = 1)
    
    training_set = lds.FeatureDataset(partition['train'], resolution, True, output_type = '2d_multiple', num_cuts = num_cuts, data_augmentation = True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size = batch_size, num_workers=2)

    val_set = lds.FeatureDataset(partition['val'], resolution, False, output_type = '2d_multiple', num_cuts = num_cuts)
    valloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, num_workers=2)
    
    test_set = lds.FeatureDataset(partition['test'], resolution, True, output_type = '2d_multiple', num_cuts = num_cuts)
    testloader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, num_workers=2)
    
    
    net2 = lmd.MCNN(net, num_cuts = num_cuts)
    
    net2 = net2.to(device)
    
    optimizer = optim.Adam(net2.parameters(), lr=learning_rate)
    
    
    
    print('\n\nStage 2: training the neural network ...')
    best_acc = 0
    
    for epoch in range(epoch2):
        
        train(epoch, net2, device, trainloader, criterion, optimizer)
        if epoch % 10 == 9:
            acc = valtest(epoch, net2, device, valloader, criterion, testval = 'Val') 
            if acc > best_acc:
                best_acc = acc
                best_net = copy.deepcopy(net2)
     
    valtest(epoch, best_net, device, testloader, criterion, testval = 'Test') 
    
            

def train_test_model(parameters):
    
    
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    
    dataset = lds.createPartition(parameters.resolution, parameters.num_train, parameters.num_val_test)
  
    
    train_test_mcnn(dataset, parameters)
