import torch
import torch.nn as nn
import torch.optim as optim
import time
from hyperparams import *
from utils import shuffle_data, squared_hinge_loss

class MNIST_Standard_MLP(nn.Module):
    """Standard PyTorch MLP for MNIST - mirrors BinaryConnect architecture"""
    
    def __init__(self, num_units=2048, n_hidden_layers=3, 
                 dropout_in=0., dropout_hidden=0., epsilon=1e-4):
        super(MNIST_Standard_MLP, self).__init__()
        
        self.dropout_in = dropout_in
        self.dropout_hidden = dropout_hidden
        
        layers = []
        
        # Input dropout
        if dropout_in > 0:
            layers.append(nn.Dropout(p=dropout_in))
        
        # Hidden layers 
        prev_units = 1 * 28 * 28  # bc01 format flattened
        for k in range(n_hidden_layers):
            # Standard linear layer (no bias when batch norm follows)
            layers.append(nn.Linear(prev_units, num_units, bias=False))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(num_units, eps=epsilon))
            
            # ReLU activation
            layers.append(nn.ReLU())
                
            # Hidden dropout
            if dropout_hidden > 0:
                layers.append(nn.Dropout(p=dropout_hidden))
                
            prev_units = num_units
        
        # Output layer (no bias when batch norm follows)
        layers.append(nn.Linear(prev_units, 10, bias=False))
                      
        # Final batch norm (no activation for output)
        layers.append(nn.BatchNorm1d(10, eps=epsilon))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # Flatten input if needed (for MNIST bc01 -> flatten)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer in self.layers:
            x = layer(x)
        return x

def train_standard_epoch(X, y, model, criterion, optimizer, batch_size, device):
    """Train one epoch for standard PyTorch model"""
    model.train()
    loss = 0
    batches = len(X) // batch_size
    
    for i in range(batches):
        batch_X = X[i*batch_size:(i+1)*batch_size].to(device)
        batch_y = y[i*batch_size:(i+1)*batch_size].to(device)
        
        optimizer.zero_grad()
        output = model(batch_X)
        batch_loss = criterion(output, batch_y)
        batch_loss.backward()
        optimizer.step()
        
        loss += batch_loss.item()
    
    loss /= batches
    return loss

def val_standard_epoch(X, y, model, criterion, batch_size, device):
    """Validate one epoch for standard PyTorch model"""
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

def train_mnist_standard(device):
    """Train standard PyTorch MNIST model"""
    # Create standard MNIST model with same architecture
    model = MNIST_Standard_MLP(
        num_units=num_units,
        n_hidden_layers=n_hidden_layers,
        dropout_in=dropout_in,
        dropout_hidden=dropout_hidden,
        epsilon=epsilon
    ).to(device)

    print(f"Standard PyTorch MNIST Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Load same data as BinaryConnect (reuse the function from mnist.py)
    from mnist import load_mnist_data
    (mnist_X_train, mnist_y_train), (mnist_X_val, mnist_y_val), (mnist_X_test, mnist_y_test) = load_mnist_data(batch_size=100)

    print(f"\nMNIST Train data: {mnist_X_train.shape}, targets: {mnist_y_train.shape}")
    print(f"MNIST Val data: {mnist_X_val.shape}, targets: {mnist_y_val.shape}")
    print(f"MNIST Test data: {mnist_X_test.shape}, targets: {mnist_y_test.shape}")
    print(f"Target format: {mnist_y_train[0]}")
    print(f"Data range: [{mnist_X_train.min():.3f}, {mnist_X_train.max():.3f}]")

    # Standard Adam optimizer (same learning rate schedule)
    optimizer = optim.Adam(model.parameters(), lr=LR_start)
    criterion = squared_hinge_loss
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_decay)
    
    # shuffle the train set
    X_train, y_train = shuffle_data(mnist_X_train, mnist_y_train)
    best_val_err = 100
    best_epoch = 1
    test_err = 100  # Initialize test error
    test_loss = 0   # Initialize test loss
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = train_standard_epoch(X_train, y_train, model, criterion, optimizer, batch_size, device)
        X_train, y_train = shuffle_data(X_train, y_train)
        
        val_err, val_loss = val_standard_epoch(mnist_X_val, mnist_y_val, model, criterion, batch_size, device)
        
        # test if validation error went down
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = epoch + 1
            test_err, test_loss = val_standard_epoch(mnist_X_test, mnist_y_test, model, criterion, batch_size, device)
        
        # Step the scheduler
        scheduler.step()
        
        epoch_duration = time.time() - start_time
        
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(current_lr))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
    
    print(f"\nFinal Standard PyTorch MNIST Results:")
    print(f"Best validation error: {best_val_err:.2f}% (epoch {best_epoch})")
    print(f"Test error: {test_err:.2f}%")
    
    return {
        'best_val_err': best_val_err,
        'test_err': test_err,
        'best_epoch': best_epoch,
        'model': model
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_mnist_standard(device)