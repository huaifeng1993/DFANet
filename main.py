from  cityscape  import DatasetTrain ,DatasetVal
import argparse
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from train import Trainer
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.dfanet import DFANet,load_backbone
from config import Config
from loss import CrossEntropyLoss2d


if __name__=='__main__':

    cfg=Config()
    #create dataset
    train_dataset = DatasetTrain(cityscapes_data_path="/raid/DataSet/Cityscape",
                                cityscapes_meta_path="/raid/DataSet/Cityscape/gtFine/")
    val_dataset = DatasetVal(cityscapes_data_path="/raid/DataSet/Cityscape",
                             cityscapes_meta_path="/raid/DataSet/Cityscape/gtFine")       
    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=16, shuffle=True,
                                           num_workers=8)
    val_loader = DataLoader(dataset=val_dataset,
                                         batch_size=16, shuffle=False, 
                                         num_workers=8)
                                         
    net = DFANet(cfg.ENCODER_CHANNEL_CFG,decoder_channel=64,num_classes=20)
    net = load_backbone(net,"backbone.pth")

    #load loss
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(
    net.parameters(), lr=0.05, momentum=0.9,weight_decay=0.00001)  #select the optimizer

    lr_fc=lambda iteration: (1-iteration/400000)**0.9

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_fc,-1)
    
    trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfg, './log')
    trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 1500)
    trainer.evaluate(val_loader)
    print('Finished Training')
