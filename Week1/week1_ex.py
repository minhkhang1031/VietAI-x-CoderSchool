
"""
Ex1: Write a program to count positive and negative numbers in a list
data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
"""
def ex1(data):
    count_pos = 0
    count_neg = 0
    for c in data:
        if c >= 0:
            count_pos += 1
        else:
            count_neg += 1

    print(f'Positive: {count_pos}')
    print(f'Negative: {count_neg}')

"""
Ex2: Given a list, extract all elements whose frequency is greater than k.
data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
k = 3
"""
def ex2(data, k):

    list = []

    for i in data:
        if i > k:
            list.append(i)

    return list

"""
Ex3: find the strongest neighbour. Given an array of N positive integers.
The task is to find the maximum for every adjacent pair in the array.
data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]
"""

def ex3(data):
    list = []

    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            list.append(data[i])
        else:
            list.append(data[i-1])

    return list

"""
Ex4: print all Possible Combinations from the three Digits
data4 = [1, 2, 3]
"""

def ex4(data):
    for i in data:
        for z in data:
            for x in data:
                if i != z and z != x and x != i:
                    print(i,x,z)


"""
# Ex5: Given two matrices (2 nested lists), the task is to write a Python program
# to add elements to each row from initial matrix.
# For example: Input : test_list1 = [[4, 3, 5,], [1, 2, 3], [3, 7, 4]], test_list2 = [[1], [9], [8]]
# Output : [[4, 3, 5, 1], [1, 2, 3, 9], [3, 7, 4, 8]]
data5_list1 = [[4, 3, 5, ], [1, 2, 3], [3, 7, 4]]
data5_list2 = [[1, 3], [9, 3, 5, 7], [8]]
"""
def ex5(data1, data2):
    for i in range(len(data1)):
        for z in range(len(data2[i])):
            data1[i].append(data2[i][z])
    return data1


"""
# Ex6:  Write a program which will find all such numbers which are divisible by 7
# but are not a multiple of 5, between 2000 and 3200 (both included).
# The numbers obtained should be printed in a comma-separated sequence on a single line.
"""

def ex6():
    for i in range(2000,3201):
        if i % 7 == 0 and i % 5 != 0:
            print(i, end=',')


"""
# Ex7: Write a program, which will find all such numbers between 1000 and 3000 (both included) such that each digit of the number is an even number.
# The numbers obtained should be printed in a comma-separated sequence on a single line.
"""

def ex7():
    for i in range(1000,3001):
        if i % 2 == 0:
            print(i, end=',')



data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
ex1(data1)
data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
k = 3
print(ex2(data2,k))
data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]
print(ex3(data3))
data4 = [1, 2, 3]
print(ex4(data4))
data5_list1 = [[4, 3, 5, ], [1, 2, 3], [3, 7, 4]]
data5_list2 = [[1, 3], [9, 3, 5, 7], [8]]
test_list1 = [[4, 3, 5,], [1, 2, 3], [3, 7, 4]]
test_list2 = [[1], [9], [8]]
print(ex5(test_list1, test_list2))
#print(ex6())
#print(ex7())

