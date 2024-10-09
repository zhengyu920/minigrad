import math
from minigrad._value import Value


def tanh(x):
    exp = math.exp(x.data * 2)
    out = (exp - 1) / (exp + 1)
    result = Value(out, prevs=[x], ops='tanh')

    def tanh_backward():
        x.grad += (1 - out ** 2) * result.grad
    result._backward = tanh_backward
    return result
