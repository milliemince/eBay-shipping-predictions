import torch
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import normalize

print("\nStarted")

dataset = np.array(pd.read_csv("/raid/cs152/eBay/eBay_ML_Challenge_Dataset_2021_train.csv"))

print("Read in dataset")



# Use only subset
subset = True
if subset:
    N = 200000
    np.random.shuffle(dataset)
    dataset = dataset[0:N + 1]
    print(f"Using subset: full N = {N}")



# Can see from full N -> N that many examples are removed...
def remove_some(dataset):
    """ Return array without weird examples (not inluding delivery days nonesense)
    """ 
    # indices of rows to delete
    delete = []
    
    for i in range(dataset.shape[0]):
        # carrier min/max missing
        if dataset[i][6] < 0 or dataset[i][7] < 0:
            delete.append(i)
            
        # zipcode invalid ("default") [something weird here, "float" not subscriptable, one zip is float? not str?]
        elif not str(dataset[i][8])[0:5].isdigit() or not str(dataset[i][9])[0:5].isdigit():
            delete.append(i)
            
        # declared_handling_days is nan
        elif dataset[i][2] != dataset[i][2]:
            delete.append(i)
            
        # weight == 0, missing or flat rate (Not sure this is how should handle...)
        elif dataset[i][15] == 0:
            delete.append(i)
    
    return np.delete(dataset, delete, 0)

data_some = remove_some(dataset)

# Shuffle data
if not subset:
    np.random.shuffle(data_some)
    print("Using full dataset")

# Percentage split 80/20
train_portion = int(0.8 * data_some.shape[0])

# Split data (partially cleaned)
train_some = data_some[:train_portion]
valid_some = data_some[train_portion:]

def make_Y(dataset):
    """ Return array of labels (includes nonesense examples) [Should use Hannah's: rounding, GMT offset, legs]
    """
    Y = []
    for entry in dataset:
        delivery = entry[14]
        payment = entry[13]

        dyear = int(delivery[0:4])
        dmonth = int(delivery[5:7])
        dday = int(delivery[8:10])

        pyear = int(payment[0:4])
        pmonth = int(payment[5:7])
        pday = int(payment[8:10])

        del_date = date(dyear, dmonth, dday)
        pay_date = date(pyear, pmonth, pday)

        difference = del_date - pay_date

        Y.append(difference.days)
    
    return np.array(Y)   

train_Y_some = make_Y(train_some)
valid_Y_some = make_Y(valid_some)

def remove_rest(dataset, Y):
    """ Return array without examples with nonesense labels
    """
    out = dataset
    delete = []
    for i in range(out.shape[0]):
        # if delivery days is negative
        if Y[i] < 0:
            delete.append(i)
            
    return np.delete(out, delete, 0)

# Split might no longer be 80/20!
train_pre = remove_rest(train_some, train_Y_some)
valid_pre = remove_rest(valid_some, valid_Y_some)

train_Y_pre = remove_rest(train_Y_some, train_Y_some)
valid_Y_pre = remove_rest(valid_Y_some, valid_Y_some)

def X2X_binary(dataset):
    """ 
    """
    out = dataset
    for entry in out:
        if entry[0] == "B2C":
            entry[0] = 1
        else:
            entry[0] = 0
            
    return out

train_pre = X2X_binary(train_pre)
valid_pre = X2X_binary(valid_pre)

def cast_zipcodes(dataset):
    """ 
    """
    out = dataset
    for i in range(dataset.shape[0]):
        # item
        out[i][8] = int(dataset[i][8][0:5])
    
        # buyer
        out[i][9] = int(dataset[i][9][0:5])
        
    return out

train_pre = cast_zipcodes(train_pre)
valid_pre = cast_zipcodes(valid_pre)

def standardize_weights(dataset):
    """ Return array with weights in lbs
    """
    out = dataset
    for i in range(dataset.shape[0]):
        if dataset[i][16] == 2:
            out[i][15] == dataset[i][15] * 2.20462
            
    return out

train_pre = standardize_weights(train_pre)
valid_pre = standardize_weights(valid_pre)

def numericize_package_size(dataset):
    """
    """
    out = dataset
    encodings = {"LETTER": 0, "PACKAGE_THICK_ENVELOPE": 1, "LARGE_ENVELOPE": 2,"VERY_LARGE_PACKAGE": 3, 
                     "LARGE_PACKAGE": 4, "EXTRA_LARGE_PACKAGE": 5, "NONE": -1}
    for i in range(dataset.shape[0]):
        out[i][17] = encodings[dataset[i][17]]
        
    return out

train_pre = numericize_package_size(train_pre)
valid_pre = numericize_package_size(valid_pre)

# Splice out unwanted features
# Excludes <acceptance_scan_timestamp> 3, <payment_datetime> 13, (<delivery_date> 14, <record_number> 18)
train_pre = train_pre[:, [0,1,2,4,5,6,7,8,9,10,11,12,15,16,17]]
valid_pre = valid_pre[:, [0,1,2,4,5,6,7,8,9,10,11,12,15,16,17]]

# Normalize data
train_pre = normalize(train_pre)
valid_pre = normalize(valid_pre)

# Convert to Tensors
train = torch.from_numpy(train_pre).float()
valid = torch.from_numpy(valid_pre).float()

train_Y = torch.from_numpy(train_Y_pre).float()
valid_Y = torch.from_numpy(valid_Y_pre).float()

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
    """ Idk how to make fancy progress bar
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
        
        if i/batch_size % 10000 == 0 and i != 0:
            print(f"Batches trained this epoch: {int(i/batch_size)}")

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
            
            if batch_count % 5000 == 0:
                print(f"Batches validated this epoch: {batch_count}")
            
        
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
learning_rate = 0.0001
gpu = True



# Main

# Use GPU if requested and available
device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
print(f"Device: {device}")



# Create model
nx = train.shape[1]
model = torch.nn.Sequential(
        torch.nn.Linear(nx, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1)
        ).to(device)



# Crit, Opt
criterion = loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Do it
print(f"\nN: {train.shape[0] + valid.shape[0]}\nbatch_size: {batch_size} \nlr: {learning_rate} \n\nmodel: {model}\n")

train_model(model, criterion, optimizer, train, train_Y, valid, valid_Y, device, num_epochs, batch_size)


