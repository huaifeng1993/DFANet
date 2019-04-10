from torchvision.models import resnet50
from thop import profile

from model.dfanet import xceptionA
model1 = xceptionA()
model2=resnet50()
#flops, params = profile(model1, input_size=(1, 3, 1024,1024))

flops, params = profile(model1, input_size=(1, 3, 1024,1024))
print(params,flops)
