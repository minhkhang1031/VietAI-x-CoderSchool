import torch

torch.manual_seed(2025)

def activation_function(x, function):
    result = None
    negative_slope = 0.01
    if function == "sigmoid":
        result = 1/(1+torch.exp(-x))
    elif function == "tanh":
        result = (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
    elif function == "relu":
        result = torch.where(x > 0, x, torch.tensor(0.0))
    elif function == "leaky_relu":
        result = torch.where(x > 0, x, negative_slope * x)
    return result

def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=1, keepdim=True)

# Define the size of each layer in the network
num_input = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_hidden_3 = 128
num_classes = 10

# Random input
input_data = torch.randn((1, num_input))
# Weights for inputs to hidden layer 1
W1 = torch.randn(num_input, num_hidden_1)
# Weights for hidden layer 1 to hidden layer 2
W2 = torch.randn(num_hidden_1, num_hidden_2)
# Weights for hidden layer 2 to hidden layer 3
W3 = torch.randn(num_hidden_2, num_hidden_3)
# Weights for hidden layer 3 to output layer
W4 = torch.randn(num_hidden_3, num_classes)

# and bias terms for hidden and output layers
B1 = torch.randn((1, num_hidden_1))
B2 = torch.randn((1, num_hidden_2))
B3 = torch.randn((1, num_hidden_3))
B4 = torch.randn((1, num_classes))

def calculate_forward(input_data, W1, B1, W2, B2, W3, B3, W4, B4):
    activation = "sigmoid"
    result = torch.matmul(input_data, W1) + B1
    result = activation_function(result, activation)
    result = torch.matmul(result, W2) + B2
    result = activation_function(result, activation)
    result = torch.matmul(result, W3) + B3
    result = activation_function(result, activation)
    result = torch.matmul(result, W4) + B4
    return softmax(result)

output = calculate_forward(input_data, W1, B1, W2, B2, W3, B3, W4, B4)
print(output)
print("Sum of probabilities:", output.sum().item())  # Should be close to 1.0

"""
--Sigmoid
tensor([[2.3007e-07, 3.3751e-08, 1.1549e-06, 4.2715e-06, 7.0812e-06, 8.9184e-10,
         9.7389e-01, 7.0983e-07, 1.9222e-10, 2.6101e-02]])
Sum of probabilities: 1.0000001192092896
--Tanh
tensor([[6.2633e-11, 2.3929e-15, 9.1485e-12, 3.5321e-09, 2.2917e-09, 3.1532e-16,
         1.0000e+00, 2.5125e-14, 1.4108e-13, 6.1036e-10]])
Sum of probabilities: 1.0
--ReLU
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
Sum of probabilities: 1.0
--Leaky ReLU
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
Sum of probabilities: 1.0
"""