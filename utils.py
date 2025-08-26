import torch
import numpy as np
from model import BinaryDenseLayer, BinaryConv2dLayer, BatchNormLayer
from hyperparams import *

def squared_hinge_loss(output, target):
    loss = torch.mean(torch.square(torch.clamp(1. - target * output, min=0)))
    return loss

def shuffle_data(X, y):
    shuffled_range = list(range(len(X)))
    np.random.shuffle(shuffled_range)
    
    new_X = X.clone()
    new_y = y.clone()
    
    for i in range(len(X)):
        new_X[i] = X[shuffled_range[i]]
        new_y[i] = y[shuffled_range[i]]
        
    return new_X, new_y

class BinaryConnectOptimizer:    
    def __init__(self, model, lr=0.001):
        self.model = model
        
        # Separate binary and non-binary parameters
        self.binary_params = []
        self.other_params = []
        
        for module in model.modules():
            if isinstance(module, BinaryDenseLayer):
                if module.binary:
                    self.binary_params.append(module.W)
                if module.b is not None:
                    self.other_params.append(module.b)
            elif isinstance(module, BatchNormLayer):
                self.other_params.extend([module.gamma, module.beta])
        
        print(f"Binary parameters: {len(self.binary_params)}")
        print(f"Other parameters: {len(self.other_params)}")
        
        # Create separate Adam optimizers
        self.binary_optimizer = torch.optim.Adam(self.binary_params, lr=lr)
        self.other_optimizer = torch.optim.Adam(self.other_params, lr=lr) if self.other_params else None
        
    def zero_grad(self):
        self.binary_optimizer.zero_grad()
        if self.other_optimizer:
            self.other_optimizer.zero_grad()
    
    def step(self, loss):
        # Step 1: Standard backward pass
        loss.backward()
        
        # Step 2: Store old parameters for W_LR_scale
        old_params = {}
        for module in self.model.modules():
            if isinstance(module, BinaryDenseLayer) and module.binary:
                old_params[id(module.W)] = module.W.data.clone()
        
        # Step 3: Update binary parameters
        self.binary_optimizer.step()
        
        # Step 4: Apply W_LR_scale and clipping
        for module in self.model.modules():
            if isinstance(module, BinaryDenseLayer) and module.binary:
                old_param = old_params[id(module.W)]
                update = module.W.data - old_param
                # Apply W_LR_scale
                module.W.data = old_param + module.W_LR_scale * update
                # Apply clipping
                module.W.data.clamp_(-module.H, module.H)
        
        # Step 5: Update other parameters normally
        if self.other_optimizer:
            self.other_optimizer.step()
    
    def set_lr(self, lr):
        for param_group in self.binary_optimizer.param_groups:
            param_group['lr'] = lr
        if self.other_optimizer:
            for param_group in self.other_optimizer.param_groups:
                param_group['lr'] = lr

def train_epoch(X, y, LR, model, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    loss = 0
    batches = len(X) // batch_size
    
    optimizer.set_lr(LR)
    
    for i in range(batches):
        batch_X = X[i*batch_size:(i+1)*batch_size].to(device)
        batch_y = y[i*batch_size:(i+1)*batch_size].to(device)
        
        optimizer.zero_grad()
        output = model(batch_X)
        batch_loss = criterion(output, batch_y)
        
        # Standard backward and update
        optimizer.step(batch_loss)
        
        loss += batch_loss.item()
    
    loss /= batches
    return loss

def val_epoch(X, y, model, criterion, device):
    """Validate one epoch"""
    model.eval()
    err = 0
    loss = 0
    batches = len(X) // batch_size
    
    with torch.no_grad():
        for i in range(batches):
            batch_X = X[i*batch_size:(i+1)*batch_size].to(device)
            batch_y = y[i*batch_size:(i+1)*batch_size].to(device)
            
            output = model(batch_X)
            batch_loss = criterion(output, batch_y)
            
            # Calculate error rate
            predicted = torch.argmax(output, dim=1)
            actual = torch.argmax(batch_y, dim=1)
            batch_err = torch.mean((predicted != actual).float())
            
            err += batch_err.item()
            loss += batch_loss.item()
    
    err = err / batches * 100
    loss /= batches
    
    return err, loss

class CIFAR10BinaryConnectOptimizer:
    def __init__(self, model, lr=0.001):
        self.model = model
        
        # Separate binary and non-binary parameters exactly like MNIST
        self.binary_params = []
        self.other_params = []
        
        for module in model.modules():
            if isinstance(module, BinaryDenseLayer):
                if module.binary:
                    self.binary_params.append(module.W)
                if module.b is not None:
                    self.other_params.append(module.b)
            elif isinstance(module, BinaryConv2dLayer):
                if module.binary:
                    self.binary_params.append(module.weight)
                if module.bias is not None:
                    self.other_params.append(module.bias)
            elif isinstance(module, BatchNormLayer):
                self.other_params.extend([module.gamma, module.beta])
        
        print(f"Binary parameters: {len(self.binary_params)}")
        print(f"Other parameters: {len(self.other_params)}")
        
        # Create separate Adam optimizers
        self.binary_optimizer = torch.optim.Adam(self.binary_params, lr=lr)
        self.other_optimizer = torch.optim.Adam(self.other_params, lr=lr) if self.other_params else None
        
    def zero_grad(self):
        self.binary_optimizer.zero_grad()
        if self.other_optimizer:
            self.other_optimizer.zero_grad()
    
    def step(self, loss):
        # Step 1: Standard backward pass (straight-through estimator in layers handles gradients)
        loss.backward()
        
        # Step 2: Store old parameters for W_LR_scale
        old_params = {}
        for module in self.model.modules():
            if isinstance(module, BinaryDenseLayer) and module.binary:
                old_params[id(module.W)] = module.W.data.clone()
            elif isinstance(module, BinaryConv2dLayer) and module.binary:
                old_params[id(module.weight)] = module.weight.data.clone()
        
        # Step 3: Update binary parameters
        self.binary_optimizer.step()
        
        # Step 4: Apply W_LR_scale and clipping
        for module in self.model.modules():
            if isinstance(module, BinaryDenseLayer) and module.binary:
                old_param = old_params[id(module.W)]
                update = module.W.data - old_param
                # Apply W_LR_scale
                module.W.data = old_param + module.W_LR_scale * update
                # Apply clipping
                module.W.data.clamp_(-module.H, module.H)
            elif isinstance(module, BinaryConv2dLayer) and module.binary:
                old_param = old_params[id(module.weight)]
                update = module.weight.data - old_param
                # Apply W_LR_scale
                module.weight.data = old_param + module.W_LR_scale * update
                # Apply clipping
                module.weight.data.clamp_(-module.H, module.H)
        
        # Step 5: Update other parameters normally
        if self.other_optimizer:
            self.other_optimizer.step()
    
    def set_lr(self, lr):
        for param_group in self.binary_optimizer.param_groups:
            param_group['lr'] = lr
        if self.other_optimizer:
            for param_group in self.other_optimizer.param_groups:
                param_group['lr'] = lr

def train_cifar10_epoch(X, y, LR, model, criterion, optimizer, batch_size, device):
    """Train one epoch for CIFAR-10"""
    model.train()
    loss = 0
    batches = len(X) // batch_size
    
    optimizer.set_lr(LR)
    
    for i in range(batches):
        batch_X = X[i*batch_size:(i+1)*batch_size].to(device)
        batch_y = y[i*batch_size:(i+1)*batch_size].to(device)
        
        optimizer.zero_grad()
        output = model(batch_X)
        batch_loss = criterion(output, batch_y)
        
        optimizer.step(batch_loss)
        
        loss += batch_loss.item()
    
    loss /= batches
    return loss

def val_cifar10_epoch(X, y, model, criterion, batch_size, device):
    """Validate CIFAR-10 using the same approach as MNIST"""
    model.eval()
    err = 0
    loss = 0
    batches = len(X) // batch_size
    
    with torch.no_grad():
        for i in range(batches):
            batch_X = X[i*batch_size:(i+1)*batch_size].to(device)
            batch_y = y[i*batch_size:(i+1)*batch_size].to(device)
            
            output = model(batch_X)
            batch_loss = criterion(output, batch_y)
            
            # Calculate error rate
            predicted = torch.argmax(output, dim=1)
            actual = torch.argmax(batch_y, dim=1)
            batch_err = torch.mean((predicted != actual).float())
            
            err += batch_err.item()
            loss += batch_loss.item()
    
    err = err / batches * 100
    loss /= batches
    
    return err, loss
