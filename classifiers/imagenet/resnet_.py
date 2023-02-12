import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torchvision.transforms as transforms

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet_(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, norm=True):
        super().__init__(block, layers)
        # super(ResNet, self).__init__(self, block=block, layers=layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
        #          groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
        #          norm_layer=norm_layer)
        # ResNet.__init__(self, block, layers, norm, num_classes=1000, zero_init_residual=False,
        #           groups=1, width_per_group=64, replace_stride_with_dilation=None,
        #           norm_layer=None)
        
        self.norm = norm
        if self.norm:
            print("[!] Your network is modified manually")
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.data_normalization = transforms.Normalize(mean, std)
    
    def forward(self, x, use_feature=False):
        features = []
        if self.norm:
            x = self.data_normalization(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features.append(x)
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if use_feature:
            return x, features
        
# =============================================================================
#         x = super().forward(x)
# =============================================================================

        return x
            

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

if __name__ == "__main__":
    model = resnet50(pretrained=True, norm=True)
    
    input = torch.randn(5, 3, 224, 224)
    output = model(input)
    
    output2, features = model(input, use_feature=True)
    
    print(output.shape)
    print(output2.shape)
    
    print(len(features))
    for f in features:
        print(f.shape)