import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from model import BinaryDenseLayer, BatchNormLayer
from hyperparams import *
from utils import *

class MNIST_MLP(nn.Module):
    """MLP for MNIST"""
    
    def __init__(self, num_units=2048, n_hidden_layers=3, binary=True, stochastic=True, 
                 H=1., dropout_in=0., dropout_hidden=0., epsilon=1e-4, alpha=.15):
        super(MNIST_MLP, self).__init__()
        
        self.binary = binary
        self.stochastic = stochastic
        self.H = H
        self.dropout_in = dropout_in
        self.dropout_hidden = dropout_hidden
        
        layers = []
        
        # Input dropout
        if dropout_in > 0:
            layers.append(nn.Dropout(p=dropout_in))
        
        # Hidden layers 
        prev_units = 1 * 28 * 28  # bc01 format flattened
        for k in range(n_hidden_layers):
            # Binary dense layer with identity nonlinearity
            layers.append(BinaryDenseLayer(
                prev_units, num_units, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale="Glorot",
                bias=False  # Original has no bias when batch norm follows
            ))
            
            # Batch normalization with ReLU nonlinearity
            layers.append(BatchNormLayer(
                num_units,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=nn.ReLU()))
                
            # Hidden dropout
            if dropout_hidden > 0:
                layers.append(nn.Dropout(p=dropout_hidden))
                
            prev_units = num_units
        
        # Output layer
        layers.append(BinaryDenseLayer(
            prev_units, 10,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale="Glorot",
            bias=False  # Original has no bias when batch norm follows
        ))
                      
        # Final batch norm with identity nonlinearity (no activation)
        layers.append(BatchNormLayer(
            10,
            epsilon=epsilon, 
            alpha=alpha,
            nonlinearity=nn.Identity()))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def train_mnist(device):
    # Create MNIST model
    model = MNIST_MLP(
        num_units=num_units,
        n_hidden_layers=n_hidden_layers,
        binary=binary,
        stochastic=stochastic,
        H=H,
        dropout_in=dropout_in,
        dropout_hidden=dropout_hidden,
        epsilon=epsilon,
        alpha=alpha
    ).to(device)

    print(f"MNIST Model created with {sum(p.numel() for p in model.parameters())} parameters")
    # print(f"Input format: bc01 (batch, channels=1, height=28, width=28)")

    (mnist_X_train, mnist_y_train), (mnist_X_val, mnist_y_val), (mnist_X_test, mnist_y_test) = load_mnist_data(batch_size=100)

    print(f"\nMNIST Train data: {mnist_X_train.shape}, targets: {mnist_y_train.shape}")
    print(f"MNIST Val data: {mnist_X_val.shape}, targets: {mnist_y_val.shape}")
    print(f"MNIST Test data: {mnist_X_test.shape}, targets: {mnist_y_test.shape}")
    print(f"Target format: {mnist_y_train[0]}")
    print(f"Data range: [{mnist_X_train.min():.3f}, {mnist_X_train.max():.3f}]")

    # Training code
    optimizer = BinaryConnectOptimizer(model, lr=LR_start)
    criterion = squared_hinge_loss
    
    # shuffle the train set
    X_train, y_train = shuffle_data(mnist_X_train, mnist_y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(X_train, y_train, LR, model, criterion, optimizer, device)
        X_train, y_train = shuffle_data(X_train, y_train)
        
        val_err, val_loss = val_epoch(mnist_X_val, mnist_y_val, model, criterion, device)
        
        # test if validation error went down
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = epoch + 1
            test_err, test_loss = val_epoch(mnist_X_test, mnist_y_test, model, criterion, device)
        
        epoch_duration = time.time() - start_time
        
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay
    
    print(f"\nFinal MNIST Results:")
    print(f"Best validation error: {best_val_err:.2f}% (epoch {best_epoch})")
    print(f"Test error: {test_err:.2f}%")

# MNIST Dataset preparation
def load_mnist_data(batch_size=100):
    """Load and preprocess MNIST dataset"""
    
    # Load datasets without any transforms first
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Extract raw data
    train_data = train_dataset.data.float() / 255.0  # Convert to [0,1]
    train_targets = train_dataset.targets
    test_data = test_dataset.data.float() / 255.0
    test_targets = test_dataset.targets
    
    # Apply centering
    mean = train_data.mean()
    train_data = train_data - mean
    test_data = test_data - mean
    
    # bc01 format 
    train_data = train_data.unsqueeze(1)  # Add channel dimension: (N, 1, 28, 28)
    test_data = test_data.unsqueeze(1)
        
    # Split training set: first 50000 for train, last 10000 for validation
    train_size = 50000
    val_size = 10000
    
    train_X = train_data[:train_size]
    train_y = train_targets[:train_size]
    val_X = train_data[train_size:train_size + val_size]
    val_y = train_targets[train_size:train_size + val_size]
    test_X = test_data
    test_y = test_targets
    
    # Convert targets to one-hot
    def to_onehot_hinge(targets, num_classes=10):
        # Onehot the targets
        targets_onehot = torch.eye(num_classes)[targets].float()
        # for hinge loss: 2 * onehot - 1
        targets_hinge = 2 * targets_onehot - 1.0
        return targets_hinge
    
    train_y = to_onehot_hinge(train_y)
    val_y = to_onehot_hinge(val_y)
    test_y = to_onehot_hinge(test_y)
    
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_mnist(device)