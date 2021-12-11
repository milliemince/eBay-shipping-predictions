# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import datetime as dt
from catboost import CatBoostRegressor, Pool

print("Imports done")

# %%
# DATA CLEANING 
# dataset = pd.read_csv('/raid/cs152/eBay/eBay_ML_Challenge_Dataset_2021_train.csv')
dataset = pd.read_csv('/raid/cs152/eBay/missing_data_dropped.csv') # , nrows=10000)

b2c_c2c = np.array(dataset["b2c_c2c"])
seller_id = np.array(dataset["seller_id"])
declared_handling_days = np.array(dataset["declared_handling_days"])
shipment_method_id = np.array(dataset["shipment_method_id"])
shipping_fee = np.array(dataset["shipping_fee"])
carrier_min_estimate = np.array(dataset["carrier_min_estimate"])
carrier_max_estimate = np.array(dataset["carrier_max_estimate"])
item_zip = dataset["item_zip"]
buyer_zip = dataset["buyer_zip"]
category_id = np.array(dataset["category_id"])
item_price = np.array(dataset["item_price"])
quantity = np.array(dataset["quantity"])
weight = np.array(dataset["weight"])
package_size = np.array(dataset["package_size"])
record_number = np.array(dataset["record_number"])
zip_distance = np.array(dataset["zip_distance"])
item_zip_pop_density = np.array(dataset["item_zip_pop_density"])
item_zip_median_income = np.array(dataset["item_zip_median_income"])
buyer_zip_pop_density = np.array(dataset["buyer_zip_pop_density"])
buyer_zip_median_income = np.array(dataset["buyer_zip_median_income"])
handling_days = np.array(dataset["handling_days"])
delivery_days = np.array(dataset["delivery_days"])

features = np.column_stack((b2c_c2c, 
    seller_id, 
    declared_handling_days,
    shipment_method_id, 
    shipping_fee,
    carrier_min_estimate,
    carrier_max_estimate, 
    item_zip, 
    buyer_zip, 
    category_id,
    item_price, 
    quantity, 
    weight, 
    package_size, 
    record_number,
    zip_distance, 
    item_zip_pop_density, 
    item_zip_median_income,
    buyer_zip_pop_density, 
    buyer_zip_median_income, 
    handling_days))

labels = np.array(delivery_days)

print("data import done")

# %%
X = features
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

# %% [markdown]
# CAT_FEATURES are the catagorical features. early_stopping_rounds tells the model to stop if it doesn't
# improve for 10 rounds (stopping overfitting). 

# %%
CAT_FEATURES = [0, 1, 3, 7, 8, 9, 12, 13, 15]
# b2c_c2c, seller_id, shipment_method_id, item_zip, buyer_zip, category_id, quantity,
# weight, package_size, zip_distance

model = CatBoostRegressor(iterations=600,
                                   cat_features = CAT_FEATURES,
                                   learning_rate = 0.375,
                                   depth = 8,
                                   l2_leaf_reg = 5,
#                                    plot=True,
                                   early_stopping_rounds = 20
                         )
print("model initialized")


# %% [markdown]
# Below is hyperparameter tuning (commented out when running model)

# %%
# grid = {'learning_rate': [0.36, 0.37, 0.375, 0.38, 0.385, 0.39, 0.4],
#         'depth': [6, 7, 8, 9, 10, 11, 12],
#         'l2_leaf_reg': [0, 1, 2, 3, 4, 5, 6]}

# randomized_search_result = model.randomized_search(grid,
#                                                    X=X_train,
#                                                    y=y_train,
#                                                    plot=True)
# print(model.get_params())

# %%
# train model
model.fit(X_train, y_train, 
                   eval_set = (X_test, y_test),
                   plot = False,
         )

# %%
print("Feature importances:")

print(model.get_feature_importance(
Pool(X_train, y_train, cat_features=CAT_FEATURES)))


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


# %% [markdown]
# Below is evaluation of the model using the loss function from eBay
#

# %%
train_score = model.score(X_train, y_train)
print("train score: " + str(train_score))
pred = model.predict(X_test)
loss = evaluate_loss(pred, y_test)
print("loss: " + str(loss))
