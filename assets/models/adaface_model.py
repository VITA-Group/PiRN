from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import Module
from torch.nn import PReLU
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import  torch.nn.functional as F

def build_model(model_name='ir_50', loc_cross_att=2, aux_feature_dim=512):

    # loc_cross_att dimension
    # 1: [4, 64, 56, 56]
    # 2: [4, 128, 28, 28]
    # 3: [4, 256, 14, 14]
    # 4: [4, 512, 7, 7])
    if model_name == 'ir_101':
        return IR_101(input_size=(112, 112), loc_cross_att=loc_cross_att, aux_feature_dim=aux_feature_dim)
    elif model_name == 'ir_50':
        return IR_50(input_size=(112, 112), loc_cross_att=loc_cross_att, aux_feature_dim=aux_feature_dim)
    elif model_name == 'ir_34':
        return IR_34(input_size=(112, 112), loc_cross_att=loc_cross_att, aux_feature_dim=aux_feature_dim)
    elif model_name == 'ir_18':
        return IR_18(input_size=(112, 112), loc_cross_att=loc_cross_att, aux_feature_dim=aux_feature_dim)
    else:
        raise ValueError('not a correct model name', model_name)


def IR_18(input_size, loc_cross_att, aux_feature_dim):
    model = Backbone(input_size, 18, 'ir', loc_cross_att, aux_feature_dim)
    return model


def IR_34(input_size, loc_cross_att, aux_feature_dim):
    model = Backbone(input_size, 34, 'ir', loc_cross_att, aux_feature_dim)
    return model


def IR_50(input_size, loc_cross_att, aux_feature_dim):
    model = Backbone(input_size, 50, 'ir', loc_cross_att, aux_feature_dim)
    return model


def IR_101(input_size, loc_cross_att, aux_feature_dim):
    model = Backbone(input_size, 100, 'ir', loc_cross_att, aux_feature_dim)
    return model


def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """ Flat tensor
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """

    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """

    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError('not implemented')

    return blocks


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, aux_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if aux_kv is not None:
            assert aux_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = aux_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum( "bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class ResidualCrossAttention(nn.Module):
    def __init__(self, n_heads=4, feature_dim=128, aux_feature_dim=512):
        super(ResidualCrossAttention, self).__init__()
        self.attention = QKVAttention(n_heads=n_heads)

        self.proj_q = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1,1))
        self.proj_k = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1,1))
        self.proj_v = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1,1))
        self.proj_aux_k = nn.Conv2d(in_channels=aux_feature_dim, out_channels=feature_dim, kernel_size=(1,1))
        self.proj_aux_v = nn.Conv2d(in_channels=aux_feature_dim, out_channels=feature_dim, kernel_size=(1,1))


    def forward(self, feature, aux_feature):
        B, C, H, W = feature.shape
        q = self.proj_q(feature).reshape(B, C, H*W)
        k = self.proj_k(feature).reshape(B, C, H*W)
        v = self.proj_v(feature).reshape(B, C, H*W)
        B, _, _H, _W = aux_feature.shape
        aux_k = self.proj_aux_k(aux_feature).reshape(B, C, _H*_W)
        aux_v = self.proj_aux_v(aux_feature).reshape(B, C, _H*_W)
        out = self.attention(torch.cat([q,k,v], dim=1), torch.cat([aux_k, aux_v], dim=1))
        out = out.reshape(B, C, H, W)
        out = out + feature
        return out


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', loc_cross_att=2, aux_feature_dim=512):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode = ir
            loc_cross_att: int between 1 and 4 (block number)
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir'], \
            "mode should be ir "
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            unit_module = BasicBlockIR
            output_channel = 512
        else:
            unit_module = BottleneckIR
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(output_channel),
                                           Dropout(0.4), Flatten(),
                                           Linear(output_channel * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

        self.block_num_mapping = {1: 2, 2: 15, 3: 45, 4: 48}
        self.block_to_channel_dim = {1: 64, 2: 128, 3: 256, 4: 512}
        self.loc_cross_att_layer = self.block_num_mapping[loc_cross_att]
        self.loc_feature_channel_dim = self.block_to_channel_dim[loc_cross_att]
        self.aux_feature_dim = aux_feature_dim
        # dimension
        # 1: [4, 64, 56, 56]
        # 2: [4, 128, 28, 28]
        # 3: [4, 256, 14, 14]
        # 4: [4, 512, 7, 7])
        self.resid_cross_att = ResidualCrossAttention(n_heads=16,
                                                      feature_dim=self.loc_feature_channel_dim,
                                                      aux_feature_dim=self.aux_feature_dim)

    def forward(self, x, auxilary_feature):

        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)
        for idx, module in enumerate(self.body):

            if idx == self.loc_cross_att_layer:
                x = self.resid_cross_att(x, auxilary_feature)
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm


if __name__ == '__main__':

    att = QKVAttention(n_heads=16)
    q = torch.randn(4, 128, 16)
    k = torch.randn(4, 128, 16)
    v = torch.randn(4, 128, 16)
    qkv = torch.cat([q,k,v], dim=1)
    aux_k = torch.randn(4, 128, 16)
    aux_v = torch.randn(4, 128, 16)
    aux_kv = torch.cat([aux_k, aux_v], dim=1)
    out = att(qkv, aux_kv).shape

