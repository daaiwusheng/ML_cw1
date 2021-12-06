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


def get_square_matrix(one_dim_array):
    dim = one_dim_array.shape[0]
    result = np.zeros((dim, dim))
    for i in range(dim):
        result[i] = one_dim_array
    return result


def get_Kronecker_delta(one_dim_array):
    dim = one_dim_array.shape[0]
    square_matrix = np.zeros((dim, dim))
    for k in range(dim):
        square_matrix[k] = one_dim_array
    identity_matrix = np.identity(dim)
    # (δκK − yK ) matrix
    delta_k_K_matrix = identity_matrix - square_matrix
    # yκ(δκK − yK ) matrix
    result = one_dim_array.reshape(dim, 1) * delta_k_K_matrix
    return result


function_soft = Softmax()
x = [0, 1, 0]

out = function_soft.forward(x)
print(out)
deta = function_soft.backward(1)
print(deta)

m = np.array([1, 2, 3])
n = [0, 0, 1, 2]
print(np.outer(m, m))

a = m.reshape(m.shape[0], 1)
b = m.reshape(1, m.shape[0])
print(a)
print(b)
print(np.dot(a, b))
identity_matrix = np.identity(get_square_matrix(m).shape[0])
sub_matrix = identity_matrix - get_square_matrix(m)
result = a * sub_matrix
print(result)
print(np.diag(m) - np.outer(m, m))
print(get_Kronecker_delta(m))
