import torch

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""
#1. 3D tensor of shape 20x30x40 with all values = 0
ex1_1 = torch.empty(20,30,40)
print(ex1_1.shape)
#2. 1D tensor containing the even numbers between 10 and 100
ex1_2 = torch.range(10,101, step=2)
print(ex1_2)
"""
    x = torch.rand(4, 6)
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
#1. Sum of all elements of x
x = torch.rand(4,6)
y = torch.sum(x)
print(x)
print(y)
# 2. Sum of the columns of x  (result is a 6-element tensor)
y_2 = torch.sum(x,0)
print(y_2)
#3. Sum of the rows of x   (result is a 4-element tensor)
y_3 = torch.sum(x,1)
print(y_3)
"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
x = torch.tensor([0.1, 0.3, 2.3, 0.45])
y = torch.tensor([0.13, 0.23, 2.33, 0.45])
cos = torch.nn.CosineSimilarity(dim=0)
ex3 = cos(x,y)
print(ex3)
"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
x_2d = torch.tensor([[0.2714, 1.1430, 1.3997, 0.8788],
                  [-2.2268, 1.9799, 1.5682, 0.5850],
                  [1.2289, 0.5043, -0.1625, 1.1403]])
y_2d = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                  [-0.6679, 0.0793, -2.5842, -1.5123],
                  [1.1110, -0.1212, 0.0324, 1.1277]])
cos_2d = torch.nn.CosineSimilarity(dim=0)
ex4 = cos_2d(x_2d,y_2d)
print(ex4)
"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
x_5 = torch.tensor([[0, 1],
                  [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9],
                  [10, 11]])
x_5 = x_5.view(12)
print(x_5.size())
x_5 = x_5.view(3,4)
print(x_5.size())
"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
x_6 = torch.rand(3, 1080, 1920)
y_6 = torch.rand(3, 720, 1280)
#1. Make x become 1x3x1080x1920 4D tensor
x_6 = x_6.view(1, 3, 1080, 1920)
print(x_6.size())
#2. Make y become 1x3x720x1280 4D tensor
y_6 = y_6.view(1, 3, 720, 1280)
print(y_6.size())
#3. Resize y to make it have the same size as x
y_6 = torch.nn.functional.interpolate(y_6, size=(1080,1920), mode='bilinear', align_corners=False)
print(y_6.size())
#4. Join them to become 2x3x1080x1920 tensor
ex_6 = torch.cat((x_6,y_6),dim=0)
print(ex_6.size())