# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2020-3-06
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class SeparableConv2d(nn.Module):
    def __init__(self, inputChannel, outputChannel, kernel_size=3, stride=1, padding=1,dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inputChannel, inputChannel, kernel_size, stride, padding, dilation,
                               groups=inputChannel, bias=bias)
        self.pointwise = nn.Conv2d(inputChannel, outputChannel, 1, 1, 0, 1, 1, bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


# encoder block
class Block(nn.Module):
    """
    Base block for XceptionA and DFANet.
    inputChannel: channels of inputs of the base block.
    outputChannel:channnels of outputs of the base block.
    stride: stride
    BatchNorm:
    """
    def __init__(self, inputChannel, outputChannel,stride=1,BatchNorm=nn.BatchNorm2d):
        super(Block, self).__init__()

        self.conv1=nn.Sequential(SeparableConv2d(inputChannel,outputChannel//4,stride=stride,),
                                BatchNorm(outputChannel//4),
                                nn.ReLU())
        self.conv2=nn.Sequential(SeparableConv2d(outputChannel//4,outputChannel//4),
                                BatchNorm(outputChannel//4),
                                nn.ReLU())
        self.conv3=nn.Sequential(SeparableConv2d(outputChannel//4,outputChannel),
                                BatchNorm(outputChannel),
                                nn.ReLU())
        self.projection=nn.Conv2d(inputChannel,outputChannel,1,stride=stride,bias=False)
        
       
    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        identity=self.projection(x)
        return out+identity


class enc(nn.Module):
    """
    encoders:
    in_channels:The channels of input feature maps
    out_channnel:the channels of outputs of this enc.
    """
    def __init__(self,in_channels,out_channels,stride=2):
        super(enc, self).__init__()
        self.block1=Block(in_channels,out_channels,stride=2)
        self.block2=Block(out_channels,out_channels)
        self.block3=Block(out_channels,out_channels)
    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        return x

class Attention(nn.Module):
    """
    self attention model.

    """
    def __init__(self,in_channels,out_channels):
        super(Attention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(in_channels,1000,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1000, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

        
class SubBranch(nn.Module):
    """
    channel_cfg: channels of the inputs of enc stage
    branch_index: 0,1,2,
    """
    def __init__(self,channel_cfg,branch_index):
        super(SubBranch,self).__init__()
        self.enc2=enc(channel_cfg[0],48)
        self.enc3=enc(channel_cfg[1],96)
        self.enc4=enc(channel_cfg[2],192)
        self.atten=Attention(192,192)
        self.branch_index=branch_index
    
    def forward(self,x0,*args):
        out0=self.enc2(x0)
        if self.branch_index in [1,2]:
            out1=self.enc3(torch.cat([out0,args[0]],1))
            out2=self.enc4(torch.cat([out1,args[1]],1))
        else:
            out1=self.enc3(out0)
            out2=self.enc4(out1)
        out3=self.atten(out2)
        return [out0,out1,out2,out3]    

class XceptionA(nn.Module):
    """
    channel_cfg:channels of inputs of enc block.
    num_classes:
    """
    def __init__(self,channel_cfg,num_classes):
        super(XceptionA, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(num_features=8),
                                nn.ReLU())
        self.branch=SubBranch(channel_cfg,branch_index=0)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
    
        self.classifier=nn.Sequential(nn.Linear(192,1000),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1000,num_classes))
        for m in self.modules():
            weight_init(m)

    def forward(self,x):
        b,c,_,_=x.size()
        x=self.conv1(x)
        _,_,_,x=self.branch(x)
        x=self.avg_pool(x).view(b,-1)
        x=self.classifier(x)
        return x

class DFA_Encoder(nn.Module):
    """
    Encoder of DFANet.
    """
    def __init__(self,channel_cfg):
        super(DFA_Encoder,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(num_features=8),
                                nn.ReLU())
        self.branch0=SubBranch(channel_cfg[0],branch_index=0)
        self.branch1=SubBranch(channel_cfg[1],branch_index=1)
        self.branch2=SubBranch(channel_cfg[2],branch_index=2)

    def forward(self,x):

        x=self.conv1(x)

        x0,x1,x2,x5=self.branch0(x)
        x3=F.interpolate(x5,scale_factor=4,mode='bilinear',align_corners=True)
        x1,x2,x3,x6=self.branch1(torch.cat([x0,x3],1),x1,x2)
        x4=F.interpolate(x6,scale_factor=4,mode='bilinear',align_corners=True)
        x2,x3,x4,x7=self.branch2(torch.cat([x1,x4],1),x2,x3)

        return [x0,x1,x2,x5,x6,x7]


class DFA_Decoder(nn.Module):
    """
        Decoder of DFANet.
    """
    def __init__(self,decode_channels,num_classes):
        super(DFA_Decoder,self).__init__()

        self.conv1=nn.Conv2d(in_channels=48,out_channels=decode_channels,kernel_size=1,bias=False)
        self.conv2=nn.Conv2d(in_channels=192,out_channels=decode_channels,kernel_size=1,bias=False)
        self.conv3=nn.Conv2d(in_channels=decode_channels,out_channels=num_classes,kernel_size=1,bias=False)

    def forward(self,x0,x1,x2,x3,x4,x5):
        x1=F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=True)
        x2=F.interpolate(x2,scale_factor=4,mode='bilinear',align_corners=True)
        x3=F.interpolate(x3,scale_factor=4,mode='bilinear',align_corners=True)
        x4=F.interpolate(x4,scale_factor=8,mode='bilinear',align_corners=True)
        x5=F.interpolate(x5,scale_factor=16,mode='bilinear',align_corners=True)

        x_shallow=self.conv1(x0+x1+x2)
        x_deep=self.conv2(x3+x4+x5)

        x=self.conv3(x_shallow+x_deep)
        x=F.interpolate(x,scale_factor=4,mode='bilinear',align_corners=True)
        return x

class DFANet(nn.Module):
    def __init__(self,channel_cfg,decoder_channel,num_classes):
        super(DFANet,self).__init__()
        self.encoder=DFA_Encoder(channel_cfg)
        self.decoder=DFA_Decoder(decoder_channel,num_classes)
        weight_init(self.encoder)
        weight_init(self.encoder)
    def forward(self,x):
        x0,x1,x2,x3,x4,x5=self.encoder(x)
        x=self.decoder(x0,x1,x2,x3,x4,x5)
        return x


def weight_init(module):
    print('initialize  ',module._get_name())
    for n,m in module.named_children():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,(nn.BatchNorm2d,nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,(nn.Sequential,
                            SubBranch,
                            enc,Block,
                            SeparableConv2d,
                            Attention,
                            DFA_Encoder,
                            DFA_Decoder)):
            weight_init(m)
        elif isinstance(m,(nn.ReLU,nn.ReLU,nn.ReLU6)):
            pass
        else:
            pass



if __name__=='__main__':
    ch_cfg=[[8,48,96],
            [240,144,288],
            [240,144,288]]

    #backbone test.
    net=XceptionA(ch_cfg[0],num_classes=19)
    input = torch.randn(4, 3, 1024, 1024)
    outputs=net(input)
    print(outputs.size())

    #decoder test
    net=DFANet(ch_cfg,64,19)
    outputs=net(input)
    print(outputs.size())