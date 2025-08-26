# Cifar10 params
cifar_batch_size = 50
cifar_num_epochs = 500
cifar_LR_start = 0.003
cifar_LR_fin = 0.000002
cifar_LR_decay = (cifar_LR_fin/cifar_LR_start)**(1./cifar_num_epochs)

# MNIST Params
num_epochs = 250
LR_start = .001
LR_fin = 0.000003
LR_decay = (LR_fin/LR_start)**(1./num_epochs)
batch_size = 100
num_units = 2048
alpha = .15
epsilon = 1e-4
n_hidden_layers = 3

# Dropout parameters
dropout_in = 0.
dropout_hidden = 0.

# BinaryConnect parameters
binary = True
stochastic = True
# (-H,+H) are the two binary values
H = 1.
# W_LR_scale = 1.    
W_LR_scale = "Glorot"