import pandas as pd
import numpy as np

dataset = np.array(pd.read_csv("/raid/cs152/eBay/eBay_ML_Challenge_Dataset_2021_quiz.csv"))

print(dataset.shape)

for row in dataset:
    print(row)
    break

X2X = False
accept_time = False
ship_id = False
carrier_min = False
carrier_max = False
item_zip = False
buyer_zip = False
pay_time = False
weight = False
weight_units = False

for row in dataset:
    
    # X2X
    # not B2C or C2C
    if row[0] != 'B2C' and row[0] != 'C2C':
        X2X = True
        
    # seller id
    
    # declared handling days
    
    # accept time
    # empty or nan
    if row[3] == "" or row[3] != row[3]:
        accept_time = True
        
    # ship method id
    # type check
    if type(row[4]) != int:
        ship_id = True
        
    # shipping fee
    
    # carrier min
    # negative, type check
    if row[6] < 0 or type(row[6]) != int:
        carrier_min = True
        print('carrier min', row[6])
    
    # carrier max
    # negative, type check
    if row[7] < 0 or type(row[6]) != int:
        carrier_max = True
        print('carrier max', row[7])
        
    # item zip
    # nan, not numeric
    try:
        if row[8] != row[8]:
            item_zip = True
            print("nan item zip")
        int(row[8][0:5])
    except:
        item_zip = True
        print('item_zip', row[8])
        
    # buyer zip
    # nan, not numeric
    try:
        if row[9] != row[9]:
            buyer_zip = True
            print("nan buyer zip")
        int(row[9][0:5])
    except:
        buyer_zip = True
        print('buyer_zip', row[9])
        
    # catagory id (maybe could use? could embed/encode?)
    
    # item price
    
    # quantity
    
    # payment time
    # empty or nan
    if row[13] == "" or row[13] != row[13]:
        pay_time = True
    
    # delivery date
    
    # weight
    # negative, type check
    if row[15] < 0 or type(row[15]) != int:
        weight = True
        
    # weigt units
    # type check
    if type(row[16]) != int:
        weight_units = True
        
    # package size?
    
    # record number
   
print(f'\n X2X {X2X}\n accept_time {accept_time}\n ship_id {ship_id}\n carrier_min {carrier_min}\n carrier_max {carrier_max}\n item_zip {item_zip}\n buyer_zip {buyer_zip}\n pay_time {pay_time}\n weight {weight}\n weight_units {weight_units}')


