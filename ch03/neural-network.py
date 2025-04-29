import numpy as np
import matplotlib.pylab as plt

def step_function(x):
  if x > 0:
    return 1
  else:
    return 0

def step_np_function(x):
  print(x)
  y = x > 0
  return y.astype(int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def softmax(x):
  max = np.max(x)
  exp_x = np.exp(x - max) # オーバーフロー対策
  sum_exp_x = np.sum(exp_x)
  y = exp_x / sum_exp_x
  return y
  
print("---step_function---")
print(step_function(3.0))

print("---step_np_function---")
print(step_np_function(np.array([1.0, -3.0, 2.0])))

print("---graph---")
x = np.arange(-5.0, 5.0, 0.1)
y = step_np_function(x)
z = sigmoid(x)
w = relu(x)
v = softmax(x)
plt.plot(x, y)
plt.plot(x, z)
plt.plot(x, w)
plt.plot(x, v)
plt.ylim(-0.1, 1.1) # y軸の範囲指定
plt.show()

print("---network---")
def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def identity_function(x):
  return x

def forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3)

  return y

network = init_network()
network_x = np.array([1.0, 0.5])
network_y = forward(network, network_x)
print(network_y)