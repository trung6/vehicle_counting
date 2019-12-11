import csv
import pandas as pd
import matplotlib.pyplot as plt

# class Reader():
    
df = pd.read_csv('traffic_measurement.csv', delimiter=',', encoding="utf-8-sig")

# ax = df.plot.bar(x='Camera_id', y='Num of vehicles', rot=0)
print(type(ax))
