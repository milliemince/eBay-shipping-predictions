# ## Generates bar graphs, histograms, and scatterplots to visualize distribution of all features

import torch
import pandas as pd 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


#returns column vector for data of feature i
def get_feature_i_data(train, i):
    return train[1:,i:i+1]

#generates bar graph of binary feature (the only binary feature in our dataset is "b2c_c2c")
def generate_binary_graph(data):
    binary = np.zeros(data.shape[0]-1)
    for point in range(1, data.shape[0]-1):
        if data[point][0][0] == "C":
            binary[point] = 1
    c = Counter(binary)
    headers = ["B2C", "C2C"]
    values = c[0], c[1]
    plt.bar(headers, values)
    plt.title("b2c_c2c")
    plt.show()


def generate_bar_graph(headers, values, title):
    plt.bar(headers, values)
    plt.title(title)
    plt.show()


def generate_histogram(data, title, bins):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()


def generate_scatterplot(data, title):
    step = 1/data.shape[0]
    x = np.arange(0, 1, step)
    y = data
    plt.scatter(x, y)
    plt.title(title)
    plt.show()


def graph_feature_i_data(train, i):
    data = get_feature_i_data(train, i)
    if i == 0:
        generate_binary_graph(data) #b2c_c2c column
    if i == 2:
        generate_histogram(data, "declared_handling_days", 10)
    if i == 4:
        headers = np.arange(0, 40)
        values = np.zeros_like(headers)
        for i in range(data.shape[0]):
            values[data[i][0]] += 1
        generate_bar_graph(headers, values,"shipment_method_id")
    if i == 5:
        generate_scatterplot(data, "shipping_fee")
    if i == 6:
        generate_histogram(data, "carrier_min_estimate", 5)
    if i == 7:
        generate_histogram(data, "carrier_max_estimate", 5)
    if i == 10:
        headers = np.arange(0, 40)
        values = np.zeros_like(headers)
        for i in range(data.shape[0]):
            values[data[i][0]] += 1
        generate_bar_graph(headers, values, "category_id")
    if i == 11:
        generate_scatterplot(data, "item_price")
    if i == 12:
        generate_histogram(data, "quantity", 5)
    if i == 15:
        generate_scatterplot(data, "weight")


train = np.array(pd.read_csv('eBay_ML_Challenge_Dataset_2021_train.csv'))

graph_feature_i_data(train, 0)
graph_feature_i_data(train, 2)
graph_feature_i_data(train, 4)
graph_feature_i_data(train, 5)
graph_feature_i_data(train, 6)
graph_feature_i_data(train, 7)
graph_feature_i_data(train, 10)
graph_feature_i_data(train, 11)
graph_feature_i_data(train, 12)
graph_feature_i_data(train, 15)
