import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Type, Any, Callable, Union, List, Optional

class CustomBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
        ):
        super(CustomBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.tfa = TFatten(inplanes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.tfa(x)
        identity = out
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #out = self.tfa(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class CustomBasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
        ):
        super(CustomBasicBlockV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.tfa = TFatten(planes)
        #self.height_block = Axial_Layer(in_channels=planes, num_heads=8, kernel_size=56, inference=False)
        #self.width_block = Axial_Layer(in_channels=planes, num_heads=8, kernel_size=56, stride=1, height_dim=False, inference=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        

        #out = self.tfa(x)
        #identity = out
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)

        out = self.tfa(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        

        return out

class CustomBasicBlockV3(nn.Module):
    expansion = 1

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
        ):
        super(CustomBasicBlockV3, self).__init__()
        # 保留原始BasicBlock的结构
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        #self.tfa = TFatten(planes)
        #self.height_block = Axial_Layer(in_channels=planes, num_heads=8, kernel_size=56, inference=False)
        #self.width_block = Axial_Layer(in_channels=planes, num_heads=8, kernel_size=56, stride=1, height_dim=False, inference=False)
        
        self.t_conv1 = nn.Sequential(
            nn.Conv1d(inplanes, inplanes, 3, padding=1),
            nn.ReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.Conv1d(inplanes, inplanes, 3, padding=1),
            nn.Sigmoid()
        )
        self.f_conv1 = nn.Sequential(
            nn.Conv1d(inplanes, inplanes, 3, padding=1),
            nn.ReLU()
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv1d(inplanes, inplanes, 3, padding=1),
            nn.Sigmoid()
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        #out = self.tfa(x)
        #identity = out
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv2(out)
        out1 = self.bn2(out1)
        
        a_t = torch.mean(out, dim=-2)#(b,c,128)
        a_f = torch.mean(out, dim=-1)#(b,c,360)
        a_t = self.t_conv1(a_t)
        a_t = self.t_conv2(a_t)
        a_t = a_t.unsqueeze(dim=-2)#(b,c,1,128)
        a_f = self.f_conv1(a_f)
        a_f = self.f_conv2(a_f)
        a_f = a_f.unsqueeze(dim=-1)#(b,c,360,1)
        a_tf = a_t * a_f#(b,c,360,128)
        out = a_tf * out1

        #out = self.tfa(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

def custom_resnet18(pretrained=False, **kwargs):
    """Constructs a modified ResNet-18 model."""
    model = models.ResNet(CustomBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'), strict=False)
    return model

def custom_resnet18v2(pretrained=False, **kwargs):
    """Constructs a modified ResNet-18 model."""
    model = models.ResNet(CustomBasicBlockV2, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'), strict=False)
    return model

def custom_resnet18v3(pretrained=False, **kwargs):
    """Constructs a modified ResNet-18 model."""
    model = models.ResNet(CustomBasicBlockV3, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'), strict=False)
    return model



class ModifiedResNet18(models.ResNet):
    def __init__(self, num_classes=1000):
        super(ModifiedResNet18, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2])
        self.num_classes = num_classes
        
        self.custom_module = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(256, num_classes)

    def _forward_impl(self, x):
        x = super(ModifiedResNet18, self)._forward_impl(x)
        
        x = x.view(x.size(0), -1)  
        x = self.custom_module(x)
        
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

        
        self.tfa = TFatten(planes)

    def forward(self, x):
        identity = x
        out = self.tfa(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = self.tfa(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_planes, planes, stride))
        #     self.in_planes = planes * block.expansion
        downsample = None
        if stride != 1 or self.in_planes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, block.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion*planes)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = block.expansion * planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        low_level_fea = self.layer1(out)
        out = self.layer2(low_level_fea)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out ,1)
        out = self.fc(out)
        return out

class TFatten(nn.Module):
    def __init__(self, in_planes):
        super(TFatten, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.t_conv1 = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, 3, padding=1),
            nn.ReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, 3, padding=1),
            nn.Sigmoid()
        )
        self.f_conv1 = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, 3, padding=1),
            nn.ReLU()
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, 3, padding=1),
            nn.Sigmoid()
        )
        self.x_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, (3, 3), padding=1),
            nn.ReLU()
        )
        # self.x_conv2 = nn.Sequential(
        #     nn.Conv2d(in_planes, in_planes, (3, 3), padding=1),
        #     nn.ReLU()
        # )
    def forward(self,x):
        # x_hat = x
        x_hat = self.x_conv1(x)
        x_hat = self.bn(x)
        # x_hat = self.x_conv2(x_hat)
        a_t = torch.mean(x, dim=-2)#(b,c,128)
        a_f = torch.mean(x, dim=-1)#(b,c,360)
        a_t = self.t_conv1(a_t)
        a_t = self.t_conv2(a_t)
        a_t = a_t.unsqueeze(dim=-2)#(b,c,1,128)
        a_f = self.f_conv1(a_f)
        a_f = self.f_conv2(a_f)
        a_f = a_f.unsqueeze(dim=-1)#(b,c,360,1)
        a_tf = a_t * a_f#(b,c,360,128)
        x_attn = a_tf * x_hat
        return x_attn

def ResNet101():
    return ResNet(BasicBlock, [3, 4, 23, 3])

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=56, stride=1, height_dim=True, inference=False):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False).to(device)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2).to(device)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3).to(device)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size).to(device)
        query_index = torch.arange(kernel_size).to(device)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            
        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x).to(device)
        kqv = self.kqv_bn(kqv) # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2, self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
        
        return output
