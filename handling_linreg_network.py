# ## MultiLayerPerceptron Regressor on *only* handling data

# First, we need to prepare data for the network. The features used in the handling network include: b2c_c2c, seller_id, declared_handling_days, acceptance_scan_timestsamp, and weight.
#
# - declared_handling_days is missing for some records 
# - payment_datetime and acceptance_scan_timestamp have [-+]HH:MM as an offset from GMT
# - weight = 0 indicates missing values
# - there are some records where dates are inconsistent (acceptance before payment, delivery before acceptance, delivery before payment 

import numpy as np
import pandas as pd
import math
import seaborn as sns
import scipy as sp
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model

full_train = pd.read_csv('eBay_ML_Challenge_Dataset_2021_train.csv', nrows=1000000)
full_test = pd.read_csv('eBay_ML_Challenge_Dataset_2021_train.csv', skiprows=1000000,nrows=100000, names=["b2c_c2c",
                                                                                                    "seller_id", "declared_handling_days",
                                                                                                    "acceptance_scan_timestamp", "shipment_method_id",
                                                                                                    "shipping_fee", "carrier_min_estimate", "carrier_max_estimate",
                                                                                                    "item_zip", "buyer_zip", "category_id", "item_price", "quantity",
                                                                                                    "payment_datetime", "delivery_date", "weight", "weight_units",
                                                                                                    "package_size", "record_number"])

full_train

# +
#separate all features into their own, individual numpy array
b2c_c2c = np.array(full_train["b2c_c2c"])
seller_id = np.array(full_train["seller_id"])
declared_handling_days = np.array(full_train["declared_handling_days"])
acceptance_scan_timestamp = np.array(full_train["acceptance_scan_timestamp"])
item_price = np.array(full_train["item_price"])
payment_datetime = np.array(full_train["payment_datetime"])
weight = np.array(full_train["weight"])
weight_units = np.array(full_train["weight_units"])

test_b2c_c2c = np.array(full_test["b2c_c2c"])
test_seller_id = np.array(full_test["seller_id"])
test_declared_handling_days = np.array(full_test["declared_handling_days"])
test_acceptance_scan_timestamp = np.array(full_test["acceptance_scan_timestamp"])
test_item_price = np.array(full_test["item_price"])
test_payment_datetime = np.array(full_test["payment_datetime"])
test_weight = np.array(full_test["weight"])
test_weight_units = np.array(full_test["weight_units"])


# +
def b2c_c2c_to_binary(arr):
    for i in range(arr.shape[0]):
        if arr[i][0] == "B":
            arr[i] = 0
        else:
            arr[i] = 1
b2c_c2c_to_binary(b2c_c2c)
b2c_c2c = np.array(b2c_c2c, dtype=int)

b2c_c2c_to_binary(test_b2c_c2c)
test_b2c_c2c = np.array(test_b2c_c2c, dtype=int)


# -

# There are multiple ways we can deal with missing data: (1) remove it, (2) fill in missing data with averages of that feature for similar training examples, (3) fill in missing data with the average of that feature across all training examples. For declared handling days, it makes sense to fill in missing data with the actual handling days, assuming that sellers are truthful and accurate about their handling days. Then we must write a function that takes the payment timestamp and acceptance scan timestamp to calculate the actual number of handling days.

# +
def round_datetime_to_date(datetime):
    days = datetime.days
    hours = datetime.seconds // 3600
    if hours > 12:
        return days + 1
    else:
        return days

def calculate_handling_days(acceptance_timestamps, payment_timestamps):
    labels = []
    for i in range(acceptance_timestamps.shape[0]):
        raw_payment = payment_timestamps[i]
        raw_acceptance = acceptance_timestamps[i]
        #parse raw_payment time string to separate year, month, date, and time
        p_year, p_month, p_date = int(raw_payment[0:4]), int(raw_payment[5:7]), int(raw_payment[8:10])
        p_hour, p_min, p_sec = int(raw_payment[11:13]), int(raw_payment[14:16]), int(raw_payment[17:19])
        p_datetime = dt.datetime(year=p_year, month=p_month, day=p_date, hour=p_hour, minute=p_min, second=p_sec)
        
        #since each acceptance timestamp and payment timestamp occur in the same area, we don't need to adjust
        #but anyways this code might be useful for the shipment network step
        
        #if raw_payment[23] == "-":
            #subtract time to modify to GMT 
            #p_datetime = p_datetime - dt.timedelta(hours=abs(int(raw_payment[23:26])))
        #else:
            #add time to modify to GMT
            #p_datetime = dt.timedelta(p_datetime) + dt.timedelta(p_offset)
            
        #parse raw_acceptance time string to separate year, month, date, and time
        raw_acceptance = acceptance_timestamps[i]
        a_year, a_month, a_date = int(raw_acceptance[0:4]), int(raw_acceptance[5:7]), int(raw_acceptance[8:10])
        a_hour, a_min, a_sec = int(raw_acceptance[11:13]), int(raw_acceptance[14:16]), int(raw_acceptance[17:19])
        a_datetime = dt.datetime(year=a_year, month=a_month, day=a_date, hour=a_hour, minute=a_min, second=a_sec)
        
        #handling days = acceptance time - payment time
        handling_days = a_datetime - p_datetime
        
        #round to nearest day
        rounded = round_datetime_to_date(handling_days)
        labels.append(rounded)
    return np.array(labels)


# -

#generate train and test labels (labels are number of handling days)
labels = calculate_handling_days(acceptance_scan_timestamp, payment_datetime)
test_labels = calculate_handling_days(test_acceptance_scan_timestamp, test_payment_datetime)

# +
#this is probably not an acceptable way to deal with missing data, but I am just replacing declared handling
#days with the actual number of handling days... as I type this I realize how that is problematic
indeces_of_nans = np.argwhere(np.isnan(declared_handling_days))
for index in indeces_of_nans:
    declared_handling_days[index] = labels[index]

indeces_of_nans = np.argwhere(np.isnan(test_declared_handling_days))
for index in indeces_of_nans:
    test_declared_handling_days[index] = labels[index]
# -

# For the time being, we will choose to replace missing weight values with the average weight of all shipments. (A way to improve this is to replace missing values with the average weight values for only products with that category ID)

avg = np.mean(weight)
weight = [weight[i] if weight[i] != 0 else avg for i in range(len(weight))]
avg = np.mean(test_weight)
test_weight = [test_weight[i] if test_weight[i] != 0 else avg for i in range(len(test_weight))]

train_features = np.column_stack((b2c_c2c, declared_handling_days, item_price, weight))
test_features = np.column_stack((test_b2c_c2c, test_declared_handling_days, test_item_price, test_weight))

labels.shape


#this is an implementation of the loss function given on evalAI's website 
def evaluate_loss(preds, actual):
    early_loss, late_loss = 0,0 
    for i in range(len(preds)):
        if preds[i] < actual[i]:
            #early shipment
            early_loss += actual[i] - preds[i]
        elif preds[i] > actual[i]:
            #late shipment
            late_loss += preds[i] - actual[i]
    loss = (1/len(preds)) * (0.4 * (early_loss) + 0.6 * (late_loss))
    return loss


print(train_features.shape)
print(labels.shape)

model = LinearRegression()
model.fit(train_features, labels)
preds = model.predict(test_features)
rounded_preds = [round(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))

# Not a bad loss for linear regression on the handling period! Let's try a univariate regression with that only considers the declared_handling_days feature

# +
train_features = declared_handling_days.reshape(-1,1)
test_features = test_declared_handling_days.reshape(-1,1)

model = LinearRegression()
model.fit(train_features, labels)
preds = model.predict(test_features)
rounded_preds = [round(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))
# -

# ## Trying with sklearn's MultiLayerPerceptron Neural Network Class

from sklearn.neural_network import MLPRegressor
network = MLPRegressor(hidden_layer_sizes = (2,3,4,3,2), activation="logistic", solver="lbfgs", max_iter=100)
network.fit(train_features, labels)

preds = network.predict(test_features)
rounded_preds = [math.floor(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))

#training network without weight feature
train_features_without_weight = train_features[:, 0:3] 
test_features_without_weight = test_features[:, 0:3]

network = MLPRegressor(hidden_layer_sizes = (2,3,4,3,2), activation="logistic", solver="lbfgs", max_iter=100)
network.fit(train_features_without_weight, labels)

preds = network.predict(test_features_without_weight)
rounded_preds = [math.floor(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))

#training network without item_price feature
train_features_without_price = train_features[:, [0,1,3]]
test_features_without_price = test_features[:, [0,1,3]]

network = MLPRegressor(hidden_layer_sizes = (2,3,4,3,2), activation="logistic", solver="lbfgs", max_iter=100)
network.fit(train_features_without_price, labels)
preds = network.predict(test_features_without_price)
rounded_preds = [math.floor(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))
