# Defining all variables
import numpy as np
#w1_list = list()
error_list = list()
inputs = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]])
output = np.array([[0],[1],[0],[1],[0],[1]])
# Weights
w1 = np.random.randn(inputs.shape[1],5)
w2 = np.random.randn(5,output.shape[1])
# Biases
b1 = 1
b2 = 1
# Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))
# # heavysidef Function
# def heavysidef(x):
#     if (x > 0):
#         y=1
#     else:
#         y=0
# return y

# Update the weights times 100.
for i in range(1000):
    # Feedforward
    z1 = np.dot(inputs,w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    error = np.sum((1/2)*(output - a2)**2)
    # Backpropagation
    ## LAYER 2
    ### derivative of the error with respect to the a2
    error_d_a2 = (a2 - output)
    ### derivative of the a2 with respect to the z2
    a2_d_z2 = a2*(1 - a2)
    ### derivative of the z2 with respect to the w2
    z2_d_w2 = a1
    ### derivative of the z2 with respect to the b2_w
    z2_d_b2 = b2
    ### delta weights 2
    delta_w2 =  error_d_a2 * a2_d_z2
    delta_w2 = np.dot(z2_d_w2.T, delta_w2)
    ### delta biases
    delta_b2 =  error_d_a2 * a2_d_z2
    delta_b2 = delta_b2 * z2_d_b2
    delta_b2 = np.sum(delta_b2)
    ### Update weights and bias
    w2 = w2 - delta_w2
    b2 = b2 - delta_b2
    ## LAYER 1
    ### derivative of the z2 with respect to the a1
    z2_d_a1 = w2
    ### derivative of the a1 with respect to the z1
    a1_d_z1 = a1*(1 - a1)
    ### derivative of the z1 with respect to the w1
    z1_d_w1 = inputs
    ### derivative of the z1 with respect to the b1_w
    z1_d_b1_w = b1
    ### delta weights 1
    delta_w1 =  error_d_a2 * a2_d_z2
    delta_w1 = np.dot(delta_w1,z2_d_a1.T)
    delta_w1 = delta_w1 * a1_d_z1
    delta_w1 = np.dot(inputs.T,delta_w1)
    ### delta bias 1
    delta_b1 =  error_d_a2 * a2_d_z2
    delta_b1 = np.dot(delta_b1,z2_d_a1.T)
    delta_b1 = delta_b1 * a1_d_z1
    delta_b1 = delta_b1 * z1_d_b1_w
    delta_b1 = np.sum(delta_b1)
    ### update w1 and b1
    w1 = w1 - delta_w1
    b1 = b1 - delta_b1
    error_list.append(error)
    print("error : ", error)
   