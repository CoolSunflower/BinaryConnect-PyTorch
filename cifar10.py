import torch
import torch.nn as nn
import numpy as np
import time
from utils import *
from model import *
from hyperparams import binary, stochastic, H, W_LR_scale, cifar_batch_size, cifar_LR_decay, cifar_LR_start, cifar_num_epochs

class CIFAR10_CNN(nn.Module):
    """
    CIFAR-10 CNN architecture
    """
    def __init__(self, binary=True, stochastic=True, H=1.0, W_LR_scale="Glorot"):
        super(CIFAR10_CNN, self).__init__()
        
        self.binary = binary
        self.stochastic = stochastic
        self.H = H
        self.W_LR_scale = W_LR_scale
        
        cifar_epsilon = 1e-4
        cifar_alpha = 0.1 
        
        # Input: 3x32x32
        
        # 128C3-128C3-P2 
        self.conv1 = BinaryConv2dLayer(3, 128, filter_size=3, padding=1, 
                                 binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale, bias=False)
        self.bn1 = BatchNormLayer(128, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        self.conv2 = BinaryConv2dLayer(128, 128, filter_size=3, padding=1,
                                 binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = BatchNormLayer(128, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        # 256C3-256C3-P2
        self.conv3 = BinaryConv2dLayer(128, 256, filter_size=3, padding=1,
                                 binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale, bias=False)
        self.bn3 = BatchNormLayer(256, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        self.conv4 = BinaryConv2dLayer(256, 256, filter_size=3, padding=1,
                                 binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = BatchNormLayer(256, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        # 512C3-512C3-P2
        self.conv5 = BinaryConv2dLayer(256, 512, filter_size=3, padding=1,
                                 binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale, bias=False)
        self.bn5 = BatchNormLayer(512, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        self.conv6 = BinaryConv2dLayer(512, 512, filter_size=3, padding=1,
                                 binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = BatchNormLayer(512, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        # 1024FP-1024FP-10FP
        self.fc1 = BinaryDenseLayer(512*4*4, 1024, bias=False,
                               binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn7 = BatchNormLayer(1024, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        self.fc2 = BinaryDenseLayer(1024, 1024, bias=False,
                               binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn8 = BatchNormLayer(1024, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.ReLU())
        
        # Output layer: 1024->10
        self.fc3 = BinaryDenseLayer(1024, 10, bias=False, 
                                   binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn_output = BatchNormLayer(10, epsilon=cifar_epsilon, alpha=cifar_alpha, nonlinearity=nn.Identity())
        
    def forward(self, x):
        # 128C3-128C3-P2  
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn2(x)
        
        # 256C3-256C3-P2
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn4(x)
        
        # 512C3-512C3-P2
        x = self.conv5(x)
        x = self.bn5(x)
        
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.bn6(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 1024FP-1024FP-10FP
        x = self.fc1(x)
        x = self.bn7(x)
        
        x = self.fc2(x) 
        x = self.bn8(x)
        
        # Output
        x = self.fc3(x)
        x = self.bn_output(x)
        
        return x

def load_cifar10_data():
    """Load CIFAR-10 data with preprocessing"""
    
    # Download CIFAR-10 data
    import torchvision
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Get raw numpy arrays
    train_data = train_dataset.data  # (50000, 32, 32, 3)
    train_labels = np.array(train_dataset.targets)
    test_data = test_dataset.data   # (10000, 32, 32, 3)
    test_labels = np.array(test_dataset.targets)
    
    # Convert to float32 and normalize to [0, 1]
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
    
    # Reshape to (N, C, H, W) format
    train_data = train_data.transpose(0, 3, 1, 2)  # (50000, 3, 32, 32)
    test_data = test_data.transpose(0, 3, 1, 2)    # (10000, 3, 32, 32)
    
    # ZCA whitening 
    def zca_whitening(X, epsilon=1e-5):
        # Flatten to (N, D) where D = C*H*W
        X_flat = X.reshape(X.shape[0], -1)
        
        # Compute mean and subtract
        mean = np.mean(X_flat, axis=0)
        X_centered = X_flat - mean
        
        # Compute covariance matrix
        cov = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
        
        # Compute ZCA transformation matrix
        U, S, _ = np.linalg.svd(cov)
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
        
        # Apply ZCA transformation
        X_whitened = np.dot(X_centered, zca_matrix.T)
        
        # Reshape back to original shape
        X_whitened = X_whitened.reshape(X.shape)
        
        return X_whitened, mean, zca_matrix
    
    # Apply ZCA whitening to training data
    train_data_whitened, mean, zca_matrix = zca_whitening(train_data)
    
    # Apply same transformations to test data
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    test_data_centered = test_data_flat - mean
    test_data_whitened = np.dot(test_data_centered, zca_matrix.T)
    test_data_whitened = test_data_whitened.reshape(test_data.shape)

    # Get val data
    val_data = train_data_whitened[45000:]
    val_labels = train_labels[45000:]
    train_data_final = train_data_whitened[:45000]
    train_labels_final = train_labels[:45000]
    
    # Convert labels to one-hot encoding
    def to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return one_hot
    
    train_labels_oh = to_one_hot(train_labels_final)
    val_labels_oh = to_one_hot(val_labels)  
    test_labels_oh = to_one_hot(test_labels)
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(train_data_final).float()
    y_train = torch.from_numpy(train_labels_oh).float()
    X_val = torch.from_numpy(val_data).float()
    y_val = torch.from_numpy(val_labels_oh).float()
    X_test = torch.from_numpy(test_data_whitened).float()
    y_test = torch.from_numpy(test_labels_oh).float()
    
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Data shape: {X_train.shape[1:]} (C, H, W)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_cifar10(device):
    model = CIFAR10_CNN(
        binary=binary,
        stochastic=stochastic, 
        H=H,
        W_LR_scale=W_LR_scale
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass with model 
    model.eval()
    test_input = torch.randn(1, 3, 32, 32).to(device)
    test_output = model(test_input)
    model.train()
    print(f"  Test forward pass: {test_input.shape} -> {test_output.shape}")

    print("Loading CIFAR-10 data...")
    cifar_train_data, cifar_val_data, cifar_test_data = load_cifar10_data()

    cifar_X_train, cifar_y_train = cifar_train_data
    cifar_X_val, cifar_y_val = cifar_val_data
    cifar_X_test, cifar_y_test = cifar_test_data

    # Convert targets to hinge format (-1/+1)
    cifar_y_train = 2 * cifar_y_train - 1.0
    cifar_y_val = 2 * cifar_y_val - 1.0  
    cifar_y_test = 2 * cifar_y_test - 1.0

    optimizer = CIFAR10BinaryConnectOptimizer(model, lr=cifar_LR_start)
    criterion = squared_hinge_loss
    
    # shuffle the train set
    X_train, y_train = shuffle_data(cifar_X_train, cifar_y_train)
    best_val_err = 100
    best_epoch = 1
    LR = cifar_LR_start

    for epoch in range(cifar_num_epochs):
        start_time = time.time()
        
        train_loss = train_cifar10_epoch(X_train, y_train, LR, model, criterion, optimizer, cifar_batch_size, device)
        X_train, y_train = shuffle_data(X_train, y_train)
        
        val_err, val_loss = val_cifar10_epoch(cifar_X_val, cifar_y_val, model, criterion, cifar_batch_size, device)
        
        # test if validation error went down
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = epoch + 1
            test_err, test_loss = val_cifar10_epoch(cifar_X_test, cifar_y_test, model, criterion, cifar_batch_size, device)
        
        epoch_duration = time.time() - start_time
        
        # Print results exactly as original
        print("Epoch "+str(epoch + 1)+" of "+str(cifar_num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  Training loss:                 "+str(train_loss))
        print("  Validation loss:               "+str(val_loss))
        print("  Validation error rate:         "+str(val_err)+"%")
        print("  Best epoch:                    "+str(best_epoch))
        print("  Best validation error rate:    "+str(best_val_err)+"%")
        print("  Test loss:                     "+str(test_loss))
        print("  Test error rate:               "+str(test_err)+"%") 
        
        LR *= cifar_LR_decay
    
    print(f"\nFinal CIFAR-10 Results:")
    print(f"Best validation error: {best_val_err:.2f}% (epoch {best_epoch})")
    print(f"Test error: {test_err:.2f}%")
