import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.utils import _single, _pair, _triple

from collections import OrderedDict, namedtuple

from itertools import islice
import operator

class Mask_Linear(torch.nn.Module):

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Mask_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._masks = OrderedDict()



        self.weight = Parameter(torch.Tensor(out_features, in_features))

        self.weight_mask = torch.ones_like(self.weight)
        self._masks['weight'] = self.weight_mask

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_mask = torch.ones_like(self.bias)
            self._masks['bias'] = self.bias_mask
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is not None:
            # return F.linear(input, self.weight.mul(self.weight_mask), self.bias.mul(self.bias_mask))
            return F.linear(input, self.weight.mul(self._masks['weight']), self.bias.mul(self._masks['bias']))
        else:
            return F.linear(input, self.weight.mul(self._masks['weight']), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def to(self, *args, **kwargs):
        super(Lenet5_sparse, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask




class _Mask_ConvNd(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_Mask_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode


        self._masks = OrderedDict()

        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_mask = torch.ones_like(self.weight)
            self._masks['weight'] = self.weight_mask
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_mask = torch.ones_like(self.weight)
            self._masks['weight'] = self.weight_mask
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_mask = torch.ones_like(self.bias)
            self._masks['bias'] = self.bias_mask
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_Mask_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask

    def to(self, *args, **kwargs):
        super(Lenet5_sparse, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)

class Mask_Conv2d(_Mask_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Mask_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            if self.bias is not None:
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                weight, self.bias.mul(self.bias_mask), self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
        if self.bias is not None:
            return F.conv2d(input, weight, self.bias.mul(self.bias_mask), self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, input):
        # print("input type:", input.type())
        # print("weight type:", self.weight.type())
        # print("mask type:", self.weight_mask.type())
        # print("_mask type:", self._masks['weight'].type())
        # return self.conv2d_forward(input, self.weight.mul(self.weight_mask))

        return self.conv2d_forward(input, self.weight.mul(self._masks['weight']))







class _Mask_NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_Mask_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self._masks = OrderedDict()

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))

            self.weight_mask = torch.ones_like(self.weight)
            self._masks['weight'] = self.weight_mask

            self.bias = Parameter(torch.Tensor(num_features))

            self.bias_mask = torch.ones_like(self.bias)
            self._masks['bias'] = self.bias_mask
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask

    def to(self, *args, **kwargs):
        super(Lenet5_sparse, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_Mask_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _Mask_BatchNorm(_Mask_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_Mask_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.affine:
            # return F.batch_norm(
            # input, self.running_mean, self.running_var, self.weight.mul(self.weight_mask), self.bias.mul(self.bias_mask),
            # self.training or not self.track_running_stats,
            # exponential_average_factor, self.eps)

            return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight.mul(self._masks['weight']), self.bias.mul(self._masks['bias']),
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

class Mask_BatchNorm2d(_Mask_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class Mask_BatchNorm1d(_Mask_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))



class Mask_ReLU(torch.nn.Module):

    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(Mask_ReLU, self).__init__()
        self.inplace = inplace
        self._masks = OrderedDict()

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask



class _MaxPoolNd(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size, stride = None,
                 padding = 0, dilation = 1,
                 return_indices = False, ceil_mode = False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        self._masks = OrderedDict()

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask

class Mask_MaxPool2d(_MaxPoolNd):

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


class _AdaptiveAvgPoolNd(nn.Module):
    __constants__ = ['output_size']

    def __init__(self, output_size):
        super(_AdaptiveAvgPoolNd, self).__init__()
        self.output_size = output_size
        self._masks = OrderedDict()

    def extra_repr(self) -> str:
        return 'output_size={}'.format(self.output_size)

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask


class Mask_AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)


class _AvgPoolNd(torch.nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad']

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )

class Mask_AvgPool2d(_AvgPoolNd):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(Mask_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self._masks = OrderedDict()

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask




class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    # p: float
    # inplace: bool

    def __init__(self, p = 0.5, inplace = False):
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

        self._masks = OrderedDict()


    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask



class Mask_Dropout(_DropoutNd):
    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)



class Mask_Sequential(torch.nn.Module):

    def __init__(self, *args):
        super(Mask_Sequential, self).__init__()
        self._masks = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)


    def __len__(self):
        return len(self._modules)


    def __dir__(self):
        keys = super(Mask_Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys


    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def masks(self, recurse=True):
        for name, mask in self.named_masks(recurse=recurse):
            yield mask



