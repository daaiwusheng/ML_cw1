import numpy as np


class Softmax(object):
    def __init__(self):
        pass

    def forward(self, x):
        self.out = np.copy(x)
        self.out -= np.max(self.out)
        self.out = np.exp(self.out)
        s = np.sum(self.out)
        self.out = self.out / s
        return self.out

    def backward(self, eta):
        print(np.diag(self.out))
        dout = np.diag(self.out) - np.outer(self.out, self.out)
        return np.dot(dout, eta)


function_soft = Softmax()
x = [0, 1, 0]

out = function_soft.forward(x)
print(out)
deta = function_soft.backward(1)
print(deta)


m = [1,0,0]
n = [0,0,1]
print(np.outer(m,n))