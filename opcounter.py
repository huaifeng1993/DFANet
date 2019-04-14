from torchvision.models import resnet50
from thop import profile

from model.dfanet import xceptionAx3
from model.backbone import xceptionA

net=xceptionAx3(num_classes=19)
#flops, params = profile(model1, input_size=(1, 3, 1024,1024))

flops, params = profile(net, input_size=(1, 3, 1024,1024))
print("params:",params,"flops:",flops)
