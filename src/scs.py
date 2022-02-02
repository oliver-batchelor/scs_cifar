import torch
import numpy as np 
import torch.nn.functional as F
import math 
from torch import nn 
import pdb

def unfold2d(x, kernel_size:int, stride:int, padding:int):
    ### using torch.nn.functional.unfold is also okay and the effiency will be compared later.

    x = F.pad(x, [padding]*4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided((bs, in_c, (h - ks) // stride + 1, (w - ks) // stride + 1, ks, ks),
        (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x

@torch.jit.script
def cos_sim_2d(x : torch.Tensor, weight : torch.Tensor, p : torch.Tensor, 
  kernel_size: int=3, stride:int =1, padding: int=1, eps: float=1e-12):

    x = unfold2d(x, kernel_size=kernel_size, stride=stride, padding=padding) # nchwkk
    n, c, h, w, _, _ = x.shape
    x = x.reshape(n,c,h,w,-1)
    x = F.normalize(x, p=2.0, dim=-1, eps=eps)

    w = F.normalize(weight, p=2.0, dim=-1, eps=eps)
    x = torch.einsum('nchwl,vcl->nvhw', x, w)
    sign = torch.sign(x)

    x = torch.abs(x) + eps
    x = x.pow(p.view(1, -1, 1, 1))
    # pdb.set_trace()
    return sign * x

class CosSim2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, stride=1,
        padding:int=0, eps=1e-12, bias=False):
        super(CosSim2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_normal_(w)
        self.w = nn.Parameter(w.view(out_channels, in_channels, -1), requires_grad=True)
        
        self.register_parameter("p", nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, 1)

    
        
    def forward(self, x):
        return cos_sim_2d(x, self.w, self.p, self.kernel_size, self.stride, self.padding, self.eps)



def scs3x3(inp, out, stride=1):
  return nn.Sequential(
    # nn.BatchNorm2d(inp),
    CosSim2d(inp, out, kernel_size=3, padding=1, stride=stride)
  )

class ScsNet(nn.Module):
  def __init__(self, num_classes=10, features=32):
    super(ScsNet, self).__init__()
        
    self.module = nn.Sequential(
      CosSim2d(3, features, kernel_size=3, padding=1),

      scs3x3(features, features),
      scs3x3(features, features*2, stride=2),

      scs3x3(features*2, features*2),
      scs3x3(features*2, features*4, stride=2),

      scs3x3(features*4, features*4),
      scs3x3(features*4, features*8, stride=2),

      scs3x3(features*8, features*8),
      scs3x3(features*8, features*8),

      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),

      # nn.BatchNorm1d(features*8),

      nn.Linear(features * 8, num_classes)
    )

  def forward(self, x):
    return self.module(x)

if __name__ == '__main__':
    layer = CosSim2d(4, 8, 7, 2, padding=3).cuda()
    data = torch.randn(10, 4, 128, 128).cuda()


    print(layer(data).shape)

    net = ScsNet().cuda()
    data = torch.randn(10, 3, 32, 32).cuda()

    print(net(data).shape)

