import numpy as np

def AND_OLD(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1 * w1 + x2 * w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1

def perceptron(x1, x2):
  x = np.array([x1, x2]) # 入力
  w = np.array([0.5, 0.5]) # 重み
  b = -0.7 # バイアス
  return np.sum(w * x) + b

def AND(x1, x2):
  x = np.array([x1, x2]) # 入力
  w = np.array([0.5, 0.5]) # 重み
  b = -0.7 # バイアス
  tmp = np.sum(w * x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def NAND(x1, x2):
  x = np.array([x1, x2]) # 入力
  w = np.array([-0.5, -0.5]) # 重み
  b = 0.7 # バイアス
  tmp = np.sum(w * x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2]) # 入力
  w = np.array([0.5, 0.5]) # 重み
  b = -0.2 # バイアス
  tmp = np.sum(w * x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

print("---AND---")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print("---XOR---")
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

print("---perceptron---")
print(perceptron(0, 0))
print(perceptron(1, 0))
print(perceptron(0, 1))
print(perceptron(1, 1))