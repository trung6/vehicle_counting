import numpy as np
l = [1,2,3,4]
# m = [1 2 3 4]
l = np.array(l)
# m = np.array(m)
print(l.shape)
# print(m.shape)
m = np.array([[1,2,3,4], [1,2,3,4]])
print(m.shape)
c = np.dot(m, l)
print(c.shape)
print(m)
# print(l)
# print(c)

m[:, [0, 1]] = m[:, [1, 0]]
m[:, [2, 3]] = m[:, [3, 2]]
print(m)
l = np.array([5, 5])
print(l.shape)
# l = np.array([[5], [5]])
# np.transpose(l)
# np.reshape(l, (1, 2))
# l = np.expand_dims(l, axis = 1)
# print(l.shape)

# m_l = np.append(m, l, axis = 1)
# print(m_l)
# from sort import Sort
# s = Sort()
x = np.array([1,2,3])
print(np.diag(x).shape)

x = np.array([1, 2, 3])
result = np.where(x < 2)
print(type(result))
y = np.array([ [1,2,3], [4,5,6], [7,8,9] ])
print(y)
# y = np.delete(y, result, axis=0)
# print(y)
# x = np.delete(x, result, axis=0)
# print(x)
# dets = np.array([ [1,2,3], [4,5,6], [7,8,9] ])
# scores = dets[:, 2]
# order = scores.argsort()[::-1]
# print(order)
print(y[ np.array([0,2]) , :])

labels = np.array([1, 2, 3])
interest = np.array([3, 8])
indices = list(np.where(np.in1d(labels, interest)))
print(len(indices))

print(np.log(0.0001))

import pandas as pd
df = pd.read_csv('traffic_measurement.csv', delimiter=',', encoding="utf-8-sig")
# cam_id,date,start_time,end_time,num_vehicles
interest_df = df.loc[lambda df: (df.date == '10/12/2019') & (df.start_time == 1), ['cam_id', 'num_vehicles']]
print(interest_df['cam_id'].tolist(), interest_df['num_vehicles'].tolist())
# print(d)
class trial():
    @staticmethod
    def trial1():
        print('je')
# trial = trial()
trial().trial1()