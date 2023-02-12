"""
Contents Independent Purifier

"""

import torch
import torch.nn as nn

import numpy as np

from utils import get_classifier
from norm import USNorm, GroupBatchNorm2d

BatchNorm2d = nn.BatchNorm2d
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Purifier_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.purifier = Purifier(norm_type=config.NORM_LIST, config=config)
        self.classifier = get_classifier(config)
        
    def add_noise(self, image):
        noise = torch.cuda.FloatTensor(*image.shape).normal_() * (16/255)
        return image + noise
         
    def forward(self, x, return_image=False):
        self.classifier.eval()
        
        # x = self.add_noise(x)
        x_tilde = self.purifier(x)
        y_pred = self.classifier(x_tilde)
        
        if return_image:
            return y_pred, x_tilde
        else:
            return y_pred


def Purifier(norm_type, config):
    global BatchNorm2d
    if type(norm_type) is list:
        BatchNorm2d = lambda num_features: USNorm(num_features, norm_type)
    elif type(norm_type) is str:
        if 'gn_' in norm_type:
            num_group = int(norm_type[norm_type.index('_')+1:])
            BatchNorm2d = lambda num_features: nn.GroupNorm(num_group, num_features)
        elif norm_type == 'bn':
            BatchNorm2d = lambda num_features: nn.BatchNorm2d(num_features)
        elif norm_type == 'in':
            BatchNorm2d = lambda num_features: nn.InstanceNorm2d(num_features)
        elif 'gbn_' in norm_type:
            num_group = int(norm_type[norm_type.index('_') + 1:])
            BatchNorm2d = lambda num_features: GroupBatchNorm2d(num_group, num_features)
        else:
            print('Wrong norm type.')
            exit()
    return PAP(config.NUM_LAYERS, first_conv=[7, 2, 3] if config.IMG_SIZE==224 else [3, 1, 1],
               norm_layer=BatchNorm2d, activation=nn.LeakyReLU, config=config)


class Dilated_Conv2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(Dilated_Conv2d, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                      padding=padding, dilation=dilation, 
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out , kernel_size=1, padding=0, bias=False),
        )
    def forward(self, x):
        return self.op(x)


class Depthwise_Separable_ConvTranspose2d(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, output_padding, kernels_per_layer):
        super(Depthwise_Separable_ConvTranspose2d, self).__init__()
        self.depthwise = nn.ConvTranspose2d(nin, nin * kernels_per_layer, 
                                            kernel_size=kernel_size, stride=stride, 
                                            padding=padding, output_padding=output_padding,
                                            groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Depthwise_Separable_Conv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, kernels_per_layer):
        super(Depthwise_Separable_Conv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, 
                                   kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )


class Random_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias,
                 conv_list=["conv", "depthwise"]):
        super(Random_Conv2d, self).__init__()

        if padding == 1:
            dilation_padding = 2
        else:
            dilation_padding = padding
        
        self.conv_layers = nn.ModuleList([])
        self.conv_list = conv_list
        for conv in conv_list:
            if conv == "conv":
                self.conv_layers.append(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                    )
            elif "group" in conv:
                groups = int(conv.split("_")[-1])
                if in_channels < groups or out_channels < groups :
                    groups = 1
                self.conv_layers.append(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, 
                              bias=bias, groups=groups)
                    )
            elif conv == "dilated":
                self.conv_layers.append(
                    Dilated_Conv2d(in_channels, out_channels, kernel_size, stride, 
                                   padding=dilation_padding, dilation=2)
                    )
            elif "depsep" in conv:
                num = int(conv.split("_")[-1])
                self.conv_layers.append(
                    Depthwise_Separable_Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride, padding=padding, kernels_per_layer=num)
                    )
            else:
                raise ValueError("Incorrect conv layer")
        
        if len(self.conv_list) == 1:
            self.conv_type = self.conv_list[0]
        else:
            self.conv_type = None

    def set_conv(self, conv_type=None):
        self.conv_type = conv_type

    def set_convs_mixed(self):
        assert isinstance(self.conv_list, list)
        self.conv_type = np.random.choice(self.conv_list)

    def forward(self, input):
        if len(self.conv_list) == 1:
            y = self.conv_layers[0](input)
        else:
            assert self.conv_type in self.conv_list
            idx = self.conv_list.index(self.conv_type)
            y = self.conv_layers[idx](input)
        return y
    
    
class Random_ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 output_padding=0, conv_list=["conv", "group_64"]):
        super(Random_ConvTranspose2d, self).__init__()
        self.conv_layers = nn.ModuleList([])
        self.conv_list = conv_list
        for conv in conv_list:
            if (conv == "conv") or (conv == "dilated"):
                self.conv_layers.append(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                       kernel_size=kernel_size, stride=stride, padding=padding, 
                                       output_padding=output_padding),
                    )
            elif "group" in conv:
                groups = int(conv.split("_")[-1])
                if in_channels < groups or out_channels < groups :
                    groups = 1
                self.conv_layers.append(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                       kernel_size=kernel_size, stride=stride, padding=padding, 
                                       output_padding=output_padding, groups=groups),
                    )
            elif "depsep" in conv:
                num = int(conv.split("_")[-1])
                self.conv_layers.append(
                    Depthwise_Separable_ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                        stride=stride, padding=padding, 
                                                        output_padding=output_padding, kernels_per_layer=num),
                    )
            else:
                raise ValueError("Incorrect conv layer")
        
        if len(self.conv_list) == 1:
            self.conv_type = self.conv_list[0]
        else:
            self.conv_type = None

    def set_conv(self, conv_type=None):
        self.conv_type = conv_type

    def set_convs_mixed(self):
        assert isinstance(self.conv_list, list)
        self.conv_type = np.random.choice(self.conv_list)

    def forward(self, input):
        if len(self.conv_list) == 1:
            y = self.conv_layers[0](input)
        else:
            assert self.conv_type in self.conv_list
            idx = self.conv_list.index(self.conv_type)
            y = self.conv_layers[idx](input)
        return y


def set_random_norm_mixed(model, block_wise=False):
    if not block_wise:
        for name, module in model.named_modules():
            if isinstance(module, USNorm):
                module.set_norms_mixed()
            if isinstance(module, Random_Conv2d):
                module.set_convs_mixed()
            if isinstance(module, Random_ConvTranspose2d):
                module.set_convs_mixed()
    else:
        for name, module in model.named_modules():
            if isinstance(module, Random_Conv2d):
                module.set_convs_mixed()
            if isinstance(module, Random_ConvTranspose2d):
                module.set_convs_mixed()
            if isinstance(module, USNorm):
                module.set_norms_mixed()
            if isinstance(module, EncoderBottleneckLayer):
                module.set_block_random_conv()
            if isinstance(module, DecoderBottleneckLayer):
                module.set_block_random_conv()
            

def set_conv(model, conv_name):
    for name, module in model.named_modules():
        if isinstance(module, USNorm):
            module.set_norms_mixed()
        if isinstance(module, Random_Conv2d):
            module.set_conv(conv_name)
        if isinstance(module, Random_ConvTranspose2d):
            module.set_conv(conv_name)


class PAP(nn.Module):
    def __init__(self, num_layers, first_conv=[7, 2, 3],
                  norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU, config=None):
        super(PAP, self).__init__()
        
        # Encoder
        self.enc_conv = nn.Sequential(
            Random_Conv2d(in_channels=3, out_channels=64, kernel_size=first_conv[0], stride=first_conv[1], 
                          padding=first_conv[2], bias=False, conv_list=config.CONV_LIST),
            norm_layer(num_features=64),
            activation(inplace=True),
        )
        
        self.enc_block1 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=128,  
                                                 layers=num_layers, downsample_method="pool",
                                                 norm_layer=norm_layer, activation=activation, config=config)
        self.enc_block2 = EncoderBottleneckBlock(in_channels=128,  hidden_channels=128, up_channels=256,  
                                                 layers=num_layers, downsample_method="conv",
                                                 norm_layer=norm_layer, activation=activation, config=config)
        self.enc_block3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=256, up_channels=512,  
                                                 layers=num_layers, downsample_method="conv",
                                                 norm_layer=norm_layer, activation=activation, config=config)
        
        # Decoder
        self.dec_block1 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=256, down_channels=256,  layers=num_layers,
                                                 norm_layer=norm_layer, activation=activation, config=config)
        self.dec_block2 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=128,  down_channels=128,   layers=num_layers,
                                                 norm_layer=norm_layer, activation=activation, config=config)
        self.dec_block3 = DecoderBottleneckBlock(in_channels=128,  hidden_channels=64,  down_channels=64,   layers=num_layers,
                                                 norm_layer=norm_layer, activation=activation, config=config)
        
        if first_conv[0] == 7:
            last_conv = Random_ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=first_conv[0], 
                                                stride=first_conv[1], padding=first_conv[2], 
                                                output_padding=1, conv_list=config.CONV_LIST)
        else:
            last_conv = Random_Conv2d(in_channels=64, out_channels=3, kernel_size=first_conv[0],
                                      stride=first_conv[1], padding=first_conv[2], 
                                      bias=False, conv_list=config.CONV_LIST)
        self.dec_conv = nn.Sequential(
            norm_layer(num_features=64),
            activation(inplace=True),
            last_conv,
        )
        self.gate = nn.Sigmoid()
        
        self.RANDOM_SKIP = config.RANDOM_SKIP
        if config.RANDOM_SKIP:
            self.random_skip1 = nn.Sequential(
                Random_Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                              bias=False, conv_list=config.CONV_LIST),
                norm_layer(num_features=64),
                activation(inplace=True),
            )
            self.random_skip2 = nn.Sequential(
                Random_Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                              bias=False, conv_list=config.CONV_LIST),
                norm_layer(num_features=128),
                activation(inplace=True),
            )
            self.random_skip3 = nn.Sequential(
                Random_Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
                              bias=False, conv_list=config.CONV_LIST),
                norm_layer(num_features=256),
                activation(inplace=True),
            )
        
        
    def forward(self, x, rand=True):
        # Encoder
        x = self.enc_conv(x)
        res1 = x
        x = self.enc_block1(x)
        res2 = x
        x = self.enc_block2(x)
        res3 = x
        x = self.enc_block3(x)
        
        # Decoder
        x = self.dec_block1(x)
        # x += self.random_skip3(res3) if self.RANDOM_SKIP else res3
        x += res3
        x = self.dec_block2(x)
        # x += self.random_skip2(res2) if self.RANDOM_SKIP else res2
        x += res2
        x = self.dec_block3(x)
        x += self.random_skip1(res1) if self.RANDOM_SKIP else res1
        
        x = self.dec_conv(x)
        x = self.gate(x)
        return x


class EncoderBottleneckBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv",
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU, config=None):
        super(EncoderBottleneckBlock, self).__init__()
        self.downsample_method = downsample_method
        
        self.layers = nn.ModuleList([])
        if downsample_method == "conv":
            for i in range(layers):
                if i == 0:
                    self.layers.append(
                        EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               up_channels=up_channels, downsample=True,
                                               norm_layer=norm_layer, activation=activation, config=config)
                        )
                else:
                    self.layers.append(
                        EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                               up_channels=up_channels, downsample=False,
                                               norm_layer=norm_layer, activation=activation, config=config)
                        )
        elif downsample_method == "pool":
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            for i in range(layers):
                if i == 0:
                    self.layers.append(
                        EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, 
                                               up_channels=up_channels, downsample=False,
                                               norm_layer=norm_layer, activation=activation, config=config)
                        )
                else:
                    self.layers.append(
                        EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, 
                                               up_channels=up_channels, downsample=False,
                                               norm_layer=norm_layer, activation=activation, config=config)
                        )
    
    def forward(self, x):
        if self.downsample_method == "pool":
            x = self.maxpool(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class DecoderBottleneckBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, down_channels, layers,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU, config=None):
        super(DecoderBottleneckBlock, self).__init__()
        
        self.num_layers = layers
        self.layers = nn.ModuleList([])
        for i in range(layers):
            if i == layers - 1:
                self.layers.append(
                    DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, 
                                           down_channels=down_channels, upsample=True,
                                           norm_layer=norm_layer, activation=activation, config=config)
                    )
            else:
                self.layers.append(
                    DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                           down_channels=in_channels, upsample=False,
                                           norm_layer=norm_layer, activation=activation, config=config)
                    )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class EncoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, up_channels, downsample,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU, config=None):
        super(EncoderBottleneckLayer, self).__init__()
        
        self.RANDOM_SKIP = config.RANDOM_SKIP
        
        if downsample:
            self.first_conv = Random_Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                                            kernel_size=3, stride=2, padding=1, bias=False,
                                            conv_list=config.CONV_LIST)
        else:
            self.first_conv= Random_Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                                           kernel_size=3, stride=1, padding=1, bias=False,
                                           conv_list=config.CONV_LIST)
        self.first_norm = norm_layer(num_features=hidden_channels)

        self.conv1 = Random_Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                   kernel_size=3, stride=1, padding=1, bias=False,
                                   conv_list=config.CONV_LIST)
        self.norm1 = norm_layer(num_features=hidden_channels)
        
        self.conv2 = Random_Conv2d(in_channels=hidden_channels, out_channels=up_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False,
                                   conv_list=config.CONV_LIST)
        self.norm2 = norm_layer(num_features=up_channels)

        if downsample:
            self.downsample = nn.Sequential(
                Random_Conv2d(in_channels=in_channels, out_channels=up_channels, 
                              kernel_size=1, stride=2, padding=0, bias=False, 
                              conv_list=config.CONV_LIST),
                norm_layer(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                Random_Conv2d(in_channels=in_channels, out_channels=up_channels, 
                              kernel_size=1, stride=1, padding=0, bias=False, 
                              conv_list=config.CONV_LIST),
                norm_layer(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None
            if self.RANDOM_SKIP:
                self.random_skip = nn.Sequential(
                    Random_Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                  kernel_size=3, stride=1, padding=1, bias=False, 
                                  conv_list=config.CONV_LIST),
                    norm_layer(num_features=in_channels),
                    activation(inplace=True),
                    )
            else:
                self.random_skip = None

        self.act = activation()
        self.conv_list = config.CONV_LIST
    
    def forward(self, x):
        identity = x
        
        x = self.first_conv(x)
        x = self.first_norm(x)
        x = self.act(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)
        elif self.random_skip is not None:
            identity = self.random_skip(identity)
            
        x = x + identity
        x = self.act(x)
        return x
    
    def set_block_random_conv(self):
        conv_name = np.random.choice(self.conv_list)
        for name, module in self.named_modules():
            if isinstance(module, Random_Conv2d):
                module.set_conv(conv_name)


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, down_channels, upsample,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU, config=None):
        super(DecoderBottleneckLayer, self).__init__()
        
        self.RANDOM_SKIP = config.RANDOM_SKIP
        
        self.first_norm = norm_layer(num_features=in_channels)
        self.first_conv = Random_Conv2d(in_channels=in_channels, out_channels=hidden_channels, 
                                        kernel_size=1, stride=1, padding=0, bias=False, 
                                        conv_list=config.CONV_LIST)
        
        self.norm1 = norm_layer(num_features=hidden_channels)
        self.conv1 = Random_Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False, 
                                   conv_list=config.CONV_LIST)
        
        self.last_norm = norm_layer(num_features=hidden_channels)
        
        if upsample:
            # self.last_conv = nn.ConvTranspose2d(in_channels=hidden_channels, 
            #                                     out_channels=down_channels, 
            #                                     kernel_size=1, stride=2, 
            #                                     output_padding=1, 
            #                                         )
            self.last_conv = Random_ConvTranspose2d(in_channels=hidden_channels, 
                                                    out_channels=down_channels, 
                                                    kernel_size=1, stride=2, 
                                                    output_padding=1, 
                                                    conv_list=config.CONV_LIST)
        else:
            self.last_conv = Random_Conv2d(in_channels=hidden_channels, out_channels=down_channels,
                                           kernel_size=1, stride=1, padding=0, bias=False, 
                                           conv_list=config.CONV_LIST)
            
        if upsample:
            self.upsample = nn.Sequential(
                norm_layer(num_features=in_channels),
                activation(inplace=True),
                # nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, 
                #                        kernel_size=1, stride=2, output_padding=1)
                Random_ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, 
                                        kernel_size=1, stride=2, output_padding=1, 
                                        conv_list=config.CONV_LIST)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                norm_layer(num_features=in_channels),
                activation(inplace=True),
                Random_Conv2d(in_channels=in_channels, out_channels=down_channels, 
                              kernel_size=1, stride=1, padding=0, bias=False,
                              conv_list=config.CONV_LIST)
            )
        else:
            self.upsample = None
            self.down_scale = None
            if self.RANDOM_SKIP:
                self.random_skip = nn.Sequential(
                    Random_Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                  kernel_size=3, stride=1, padding=1, bias=False,
                                  conv_list=config.CONV_LIST),
                    norm_layer(num_features=in_channels),
                    activation(inplace=True),
                    )
            else:
                self.random_skip = None
            
        self.act = activation()
        self.conv_list = config.CONV_LIST
    
    def forward(self, x):
        identity = x
        
        x = self.first_norm(x)
        x = self.act(x)
        x = self.first_conv(x)
        
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
            
        x = self.last_norm(x)
        x = self.act(x)
        x = self.last_conv(x)
        
        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)
        elif self.random_skip is not None:
            identity = self.random_skip(identity)
            
        x = x + identity
        return x
    
    def set_block_random_conv(self):
        conv_name = np.random.choice(self.conv_list)
        for name, module in self.named_modules():
            if isinstance(module, Random_Conv2d):
                module.set_conv(conv_name)
            if isinstance(module, Random_ConvTranspose2d):
                module.set_conv(conv_name)

    
