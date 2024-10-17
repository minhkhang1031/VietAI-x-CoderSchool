import numpy as np
import re
"""
# Ex1: Write a NumPy program to reverse an array (first element becomes last).
# Input: [12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]
"""
data1 = np.arange(12,38)[::-1]
print(data1)

"""
# Ex2: Write a NumPy program to test whether each element of a 1-D array is also present in a second array
# Input Array1: [ 0 10 20 40 60]
#       Array2: [10, 30, 40]
"""
data2_1 = np.array([0,10,20,40,60])
data2_2 = np.array([10,30,40])
data2 = np.array([i for i in data2_1 if i in data2_2])
print(data2)

"""
# Ex3: Write a NumPy program to find the indices of the maximum and minimum values along the given axis of an array
# Input Array [1,6,4,8,9,-4,-2,11]
"""

data3 = np.array([1,6,4,8,9,-4,-2,11])
max = np.max(data3)
min = np.min(data3)

print(f'Max Array is: {max}\n'
      f'Min Array is: {min}')

"""
# Ex4: Read the entire file story.txt and write a program to print out top 100 words occur most
# frequently and their corresponding appearance. You could ignore all
# punction characters such as comma, dot, semicolon, ...
# Sample output:
# house: 453
# dog: 440
# people: 312
# ...
"""

with open('story.txt', 'r', encoding='utf-8-sig') as file:
    data = file.read()

#arr = re.split(r'[,.;:?!() \n\\\'"]',data)
arr = re.findall(r'\b\w+\b', data)
arr = [i.strip() for i in arr if i.strip()]
items, quantity = np.unique(arr, return_counts=True)

rs = sorted(zip(items, quantity), key=lambda x: x[1], reverse=True)

for w, q in rs[:100]:
      print(f'{w}: {q}')


