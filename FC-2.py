import torch
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import normalize

print("\nStarted")

dataset_full = np.array(pd.read_csv("/raid/cs152/eBay/missing_data_dropped.csv"))

print("Read in dataset, not shuffling")

# Use only subset
subset = True
if subset:
    N = 1000000
    # np.random.shuffle(dataset_full)
    dataset = dataset_full[0:N]
    print(f"Using subset: full N = {N}")

# Shuffle data
if not subset:
    np.random.shuffle(dataset_full)
    dataset = dataset_full
    print("Using full dataset")

# Percentage split 80/20
train_portion = int(0.8 * dataset.shape[0])

# Split data (partially cleaned)
train = dataset[:train_portion]
valid = dataset[train_portion:]

# remove rows w/ bad labels
def remove_bad_labels(train, valid):
    num_bad = 0
    delete_train = []
    for i in range(train.shape[0]):
        if train[i][23] < 0:
            delete_train.append(i)
            num_bad += 1
            
    delete_valid = []
    for i in range(valid.shape[0]):
        if valid[i][23] < 0:
            delete_valid.append(i)
            num_bad += 1
    
    print("num bad: " + str(num_bad))
    return np.delete(train, delete_train, 0), np.delete(valid, delete_valid, 0)

train_c, valid_c = remove_bad_labels(train, valid)

# Splice out unwanted features, but has labels
train_s = train_c[:, [7,8,14,17,22,23]]
valid_s = valid_c[:, [7,8,14,17,22,23]]
# train_s = train_c[:, [2,6,7,8,12,14,17,18,19,20,21,22,23]]
# valid_s = valid_c[:, [2,6,7,8,12,14,17,18,19,20,21,22,23]]

# remove rows w/ nan
def remove_nan(train, valid):
    num_bad = 0
    delete_train = []
    for i in range(train.shape[0]):
        if np.isnan(np.array(train[i].tolist())).any(axis=0):
            delete_train.append(i)
            num_bad += 1
            
    delete_valid = []
    for i in range(valid.shape[0]):
        if np.isnan(np.array(valid[i].tolist())).any(axis=0):
            delete_valid.append(i)
            num_bad += 1
    
    print("num bad: " + str(num_bad))
    return np.delete(train, delete_train, 0), np.delete(valid, delete_valid, 0)

train_sc, valid_sc = remove_nan(train_s, valid_s)

# get labels
train_y = train_sc[:, [train_sc.shape[1] - 1]]
valid_y = valid_sc[:, [valid_sc.shape[1] - 1]]

# splice labels from data
train_ssc = train_sc[:, :train_sc.shape[1] - 1]
valid_ssc = valid_sc[:, :valid_sc.shape[1] - 1]

# Normalize data
train_n = normalize(train_ssc)
valid_n = normalize(valid_ssc)

# Convert to Tensors
train_t = torch.from_numpy(train_n).float()
valid_t = torch.from_numpy(valid_n).float()

train_y = torch.tensor(train_y[:, 0].tolist()).float()
valid_y = torch.tensor(valid_y[:, 0].tolist()).float()

print("Tensored datasets")

def loss_function(Yhat, Y):
    """ eBay's criterion
    """
    early_loss, late_loss = 0, 0 
    
    for i in range(len(Yhat)):
        # early
        if Yhat[i] < Y[i]:
            early_loss += Y[i] - Yhat[i]
        
        # late
        elif Yhat[i] > Y[i]:
            late_loss += Yhat[i] - Y[i]
    loss = (1/len(Yhat)) * (0.4 * (early_loss) + 0.6 * (late_loss))
    
    return loss

def train_one_epoch(X, Y, batch_size, model, criterion, optimizer, device):
    """
    """
    model.train()
    
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size].to(device)
        Y_batch = Y[i:i + batch_size].to(device)
        
        output = model(X_batch)
        
        loss = criterion(output, Y_batch)
        
        model.zero_grad()
        
        loss.backward()
        
        optimizer.step()

def validate(X, Y, batch_size, model, criterion, device, epoch, num_epochs):
    """
    """
    model.eval()
    
    loss = 0
    
    batch_count = 0
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size].to(device)
            Y_batch = Y[i:i + batch_size].to(device)

            output = model(X_batch)
            
            loss += criterion(output, Y_batch).item()
            
            batch_count += 1            
        
        loss /= batch_count
    
    message = "Initial " if epoch == 0 else f"Epoch {epoch:>2}/{num_epochs}: "
    message += f"loss={loss:.3f}"
    print(message)

def train_model(model, criterion, optimizer, train, train_Y, valid, valid_Y, device, num_epochs, batch_size):
    validate(valid, valid_Y, batch_size, model, criterion, device, 0, num_epochs)
    
    for epoch in range(num_epochs):
        train_one_epoch(train, train_Y, batch_size, model, criterion, optimizer, device)
        validate(valid, valid_Y, batch_size, model, criterion, device, epoch + 1, num_epochs)

# Hyperparams
num_epochs = 100
batch_size = 128
learning_rate = 0.001
gpu = False

# Main

# Use GPU if requested and available
device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create model
nx = train_t.shape[1]
model = torch.nn.Sequential(
        torch.nn.Linear(nx, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1)
        ).to(device)

# Crit, Opt
criterion = loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Do it
print(f"\nN train: {train_t.shape[0]}\nN valid: {valid_t.shape[0]}\nbatch_size: {batch_size} \nlr: {learning_rate} \n\nmodel: {model}\n")

train_model(model, criterion, optimizer, train_t, train_y, valid_t, valid_y, device, num_epochs, batch_size)

