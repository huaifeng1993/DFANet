from  cityscape  import DatasetTrain ,DatasetVal
import argparse
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from train import Trainer
from criterion import CriterionDSN, CriterionOhemDSN
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.dfanet import xceptionAx3
from config import Config

#from pspnet import Res_Deeplab

if __name__=='__main__':

    cfg=Config()
    #create dataset
    train_dataset = DatasetTrain(cityscapes_data_path="/home/shen/Data/DataSet/Cityscape",
                                cityscapes_meta_path="/home/shen/Data/DataSet/Cityscape/gtFine/")
    val_dataset = DatasetVal(cityscapes_data_path="/home/shen/Data/DataSet/Cityscape",
                             cityscapes_meta_path="/home/shen/Data/DataSet/Cityscape/gtFine")
    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=12, shuffle=True,
                                           num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                                         batch_size=24, shuffle=False, 
                                         num_workers=4)
   
    net = xceptionAx3(num_classes=20)
  
    #load loss
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(
    net.parameters(), lr=0.5, momentum=0.9,weight_decay=0.00001)  #select the optimizer
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


    trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfg, './log')
    #trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 1000)
    #trainer.evaluate(valid_loader)
    print('Finished Training')
