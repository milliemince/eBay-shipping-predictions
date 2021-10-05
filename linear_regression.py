# %% [markdown]
# ## Naive Linear Regression

# %% [markdown]
# 3 naive linear regression models are presented: (1) normal multivate linear regression; (2) Ridge regression; (3) Lasso Regression. (1) and (2) both perform with a similar loss rate ~0.82 (not good, because the random baseline loss is 0.75!). (3) performs worse than (1) and (2), possibly because Lasso regression works better with a much larger feature space. We hypothesize that these naive linear models perform poorly because they make the assumption that all our features are drawn from Gaussian distributions, which is not the case.

# %%
import numpy as np
import pandas as pd
import torch
from datetime import date
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model

# %%
#tried to use quiz.csv as test data-set, but labels are not given; separated train in an 80/20 split
full_train = np.array(pd.read_csv('eBay_ML_Challenge_Dataset_2021_train.csv', skiprows=3000000))
full_test = np.array(pd.read_csv('eBay_ML_Challenge_Dataset_2021_train.csv', nrows=3000000))

# %%
full_train.shape, full_test.shape

# %% [markdown]
# Because this naive model assumes that features are drawn from continuous, normally distributed spaces, we will splice out all data EXCEPT b2c_c2c (column 0), shipping_fee (column 5), carrier_min_estimate (column 6), carrier_max_estimate (column 7), category_id (column 10), item_price (column 11), quantity (column 12), weight (column 15). Other columns involve unique identifiers like seller_id, zip_code, etc. that we have not decided how to incorporate.

# %%
#splice out data header and unwanted columns
train = full_train[1:, [0, 5, 6, 7, 10, 11, 12, 15]]
test = full_test[1:, [0, 5, 6, 7, 10, 11, 12, 15]]

# %%
train.shape, test.shape


# %%
#business to consumer vs. consumer to consumer feature is currently represented by strings "B2C" or "C2C"
#this function changes those strings to binary integer values: [0,1]
def b2c_c2c_to_binary(train):
    for i in range(train.shape[0]):
        if train[i][0][0] == "B":
            train[i][0] = 0
        else:
            train[i][0] = 1


# %%
b2c_c2c_to_binary(train)
b2c_c2c_to_binary(test)

# %%
#check that features in b2c_c2c column are integers now
train[:, 0], test[:, 0]


# %% [markdown]
# This dataset does not give direct labels (number of days for shipment). Rather, they give timestamps for the time of purchase and the date of delivery. We use the date class from datetime to easily subtract two dates for the number of days between them.

# %%
def calculate_labels(data):
    labels = []
    delivery_date = data[:, 14]
    payment_date = data[:, 13]
    for i in range(data.shape[0]):
        delivery = delivery_date[i]
        d_year, d_month, d_date = int(delivery[0:4]), int(delivery[5:7]), int(delivery[8:10])
        payment = payment_date[i]
        p_year, p_month, p_date = int(payment[0:4]), int(payment[5:7]), int(payment[8:10])
        date_of_delivery = date(d_year, d_month, d_date)
        date_of_payment = date(p_year, p_month, p_date)
        difference = date_of_delivery - date_of_payment
        labels.append(difference.days)
    return np.array(labels)


# %%
#create train and test labels, splicing out header
train_labels = calculate_labels(full_train)
train_labels = train_labels[1:]

test_labels = calculate_labels(full_test)
test_labels = test_labels[1:]

# %%
train_labels.shape, test_labels.shape

# %%
#normalize data for training
normalize(train)
normalize(test)


# %% [markdown]
# We need a loss function to evaluate how well our model is performing. Fortunately, the competition provides the loss function that they will use to test our model; it is implemented below

# %%
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


# %%
model = LinearRegression()
model.fit(train, train_labels)
preds = model.predict(test)
rounded_preds = [round(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))

# %%
ridge_model = Ridge(alpha=0.15)
ridge_model.fit(train, train_labels)
preds = ridge_model.predict(test)
rounded_preds = [round(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))

# %%
lasso_model = linear_model.Lasso(alpha=1.0)
lasso_model.fit(train, train_labels)
preds = lasso_model.predict(test)
rounded_preds = [round(pred) for pred in preds]
print("Loss: " + str(evaluate_loss(rounded_preds, test_labels)))
