import numpy as np
import plotly
from plotly.graph_objs import Scatter, Layout

#----------------Video Position 15:31
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

x = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
y = np.array([0,1,1])
W = np.random.rand(len(y), len(x))

# in this case x are the images
# y are the class labels and W are the weights
loss_i = L_i_vectorized(x, y, W)
print(loss_i)

#----------------Video Position 23:05
#L2 Regularization
w_1 = np.array([1,0,0,0])
w_2 = np.array([0.25,0.25,0.25,0.25])

def L_2_Regularization(W):
    return np.sum(np.square(W))

def L_1_Regularization(W):
    return np.sum(np.abs(W))

#----------------Video Position 31:24
#softmax
def softmax(s,k):
    # s = scores
    # k = correct class
    return -np.log10(np.exp(s[k])/(np.sum(np.exp(s))))
softmax([3.2,5.1,-1.7],1)

def svm_loss(s,k):
    margins = np.maximum(0, s - s[k] + 1)
    margins[k] = 0
    loss_i = np.sum(margins)
    return loss_i

# Scatter points
#LOG10 Plot
x = np.linspace(10,-10,100)
y = np.log10(x)
trace = go.Scatter(x = x, y = y)
plotly.offline.plot([trace])

#---------------Video Position 38:22
s_1 = np.array([10,-2,3])
s_2 = np.array([10,9,9])
s_3 = np.array([10,-100,-100])
print(softmax(s_1,0))
print(softmax(s_2,0))
print(softmax(s_3,0))

print(svm_loss(s_1,1))
print(svm_loss(s_2,1))
print(svm_loss(s_3,1))

#---------------Video Position 53:59
W = np.array([0.34,-1.11,0.78,0.12,0.55,2.81,-3.1,-1.5,0.33])
