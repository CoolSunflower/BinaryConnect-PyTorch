import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hard_sigmoid(x):
    return torch.clamp((x+1.)/2.,0,1)

def binarization(W, H, binary=True, deterministic=False, stochastic=False, srng=None):    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        Wb = W    
    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        
        # Stochastic BinaryConnect
        if stochastic:        
            Wb = torch.bernoulli(Wb)
        # Deterministic BinaryConnect (round to nearest)
        else:
            Wb = torch.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = torch.where(Wb==1, H, -H)
    
    return Wb

# This class extends PyTorch Linear to support BinaryConnect 
class BinaryDenseLayer(nn.Module):
    
    def __init__(self, in_features, out_features, 
                 binary=True, stochastic=True, H=1., W_LR_scale="Glorot", bias=True):
        
        super(BinaryDenseLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.binary = binary
        self.stochastic = stochastic
        
        # H parameter
        self.H = H
        if H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (in_features + out_features)))
            
        # W_LR_scale 
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (in_features + out_features)))
        
        # Initialize weights
        if self.binary:
            self.W = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-self.H, self.H))
            # Mark as binary parameter
            self.W.binary = True
        else:
            self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
            self.W.binary = False
            
        if bias:
            self.b = nn.Parameter(torch.zeros(out_features))
            self.b.binary = False
        else:
            self.register_parameter('b', None)
        
        # For deterministic/stochastic switching
        self.Wb = None
            
    def forward(self, input):
        # Flatten input if needed (for MNIST bc01 -> flatten)
        if input.dim() > 2:
            input = input.view(input.size(0), -1)
        
        # Binarize weights
        if self.binary:
            # Forward pass: use binarized weights
            Wb = binarization(self.W, self.H, self.binary, 
                            deterministic=not self.training, 
                            stochastic=self.stochastic)
            # Backward pass: gradient flows through original weights W
            Wb = Wb.detach() + self.W - self.W.detach()
        else:
            Wb = self.W
        
        # Use the weights for forward pass
        output = F.linear(input, Wb, self.b)
        
        return output
    
class BinaryConv2dLayer(nn.Module):    
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", bias=True):
        
        super(BinaryConv2dLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size if isinstance(filter_size, tuple) else (filter_size, filter_size)
        self.stride = stride
        self.padding = padding
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(self.filter_size)*in_channels)
            num_units = int(np.prod(self.filter_size)*out_channels)
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(self.filter_size)*in_channels)
            num_units = int(np.prod(self.filter_size)*out_channels)
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            
        self._srng = torch.Generator()
        self._srng.manual_seed(np.random.randint(1, 2147462579))
            
        if self.binary:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.filter_size))
            nn.init.uniform_(self.weight, -self.H, self.H)
            self.weight.binary = True
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.filter_size))
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
            self.weight.binary = False
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            self.bias.binary = False
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        if self.binary:
            # Forward pass: use binarized weights
            Wb = binarization(self.weight, self.H, self.binary, 
                            deterministic=not self.training, 
                            stochastic=self.stochastic, 
                            srng=self._srng)
            # Backward pass: gradient flows through original weights
            Wb = Wb.detach() + self.weight - self.weight.detach()
        else:
            Wb = self.weight
        
        # Use the weights for forward pass
        rvalue = F.conv2d(input, Wb, self.bias, self.stride, self.padding)
        
        return rvalue
    
class BatchNormLayer(nn.Module):
    def __init__(self, num_features, axes=None, epsilon=0.01, alpha=0.5,
            nonlinearity=None):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).
        
        @param num_features: number of features in input
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__()
        
        self.num_features = num_features
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinearity
            
        self.mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.std = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.ones(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))

    def forward(self, input):
        if not self.training:  # deterministic
            mean = self.running_mean
            std = self.running_std
        else:
            # use this batch's mean and std
            if input.dim() == 2:  # Dense layer
                mean = input.mean(0, keepdim=False)
                std = input.std(0, keepdim=False, unbiased=False)
                # and update the stored mean and std:
                with torch.no_grad():
                    self.running_mean.copy_((1 - self.alpha) * self.running_mean + self.alpha * mean)
                    self.running_std.copy_((1 - self.alpha) * self.running_std + self.alpha * std)
            elif input.dim() == 4:  # Conv layer
                mean = input.mean([0, 2, 3], keepdim=False)
                std = input.std([0, 2, 3], keepdim=False, unbiased=False)
                # and update the stored mean and std:
                with torch.no_grad():
                    self.running_mean.copy_((1 - self.alpha) * self.running_mean + self.alpha * mean)
                    self.running_std.copy_((1 - self.alpha) * self.running_std + self.alpha * std)
            else:
                raise ValueError(f"Expected 2D or 4D input, got {input.dim()}D")
                
        std = std + self.epsilon
        
        # Normalize and apply gamma/beta
        if input.dim() == 2:
            normalized = (input - mean) * (self.gamma / std) + self.beta
        elif input.dim() == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
            normalized = (input - mean) * (gamma / std) + beta
            
        return self.nonlinearity(normalized)
