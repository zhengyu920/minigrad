import math


class Value:
    """
    A class representing a scalar value that supports basic arithmetic operations
    with automatic differentiation (backpropagation).

    Attributes:
        data (float or int): The scalar value.
        prevs (list): List of previous `Value` objects used in operations leading to this value.
        ops (str): The operation that generated this value (e.g., '+', '*', '**', etc.).
        grad (float): The gradient for backpropagation.
        label (str): Optional label for the value, for easier debugging or identification.
        _backward (function): The backward pass function for computing gradients.
    """

    def __init__(self, data, prevs=[], ops='', label=''):
        self.data = data
        self.prevs = prevs
        self.ops = ops
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return str(f'Value(data: {self.data})')

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, [self, other], '+')

        def add_backward():
            self.grad += 1.0 * result.grad
            other.grad += 1.0 * result.grad
        result._backward = add_backward
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data - other.data, [self, other], '-')

        def sub_backward():
            self.grad += 1.0 * result.grad
            other.grad += -1.0 * result.grad
        result._backward = sub_backward
        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, [self, other], '*')

        def mul_backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = mul_backward

        return result

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other, modulo=None):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(math.pow(self.data, other.data), [self, other], '**')

        def pow_backward():
            self.grad += other.data * \
                math.pow(self.data, other.data - 1) * result.grad
        result._backward = pow_backward

        return result

    def __radd__(self, other):
        return Value(other) + self

    def __rsub__(self, other):
        return Value(other) - self

    def __rmul__(self, other):
        return Value(other) * self

    def __rtruediv__(self, other):
        return Value(other) / self

    def __rpow__(self, other):
        return Value(other) ** self

    def _zero_grad(self):
        self.grad = 0

    def exp(self):
        exp = math.exp(self.data)
        result = Value(exp, [self], 'exp')

        def exp_backward():
            self.grad += exp * result.grad
        result._backward = exp_backward

        return result

    def backward(self):
        self.grad = 1.0
        visited = set()
        topo = []

        def DFS(cur):
            if cur in visited:
                return
            visited.add(cur)
            for prev in cur.prevs:
                DFS(prev)
            topo.append(cur)

        DFS(self)
        for node in reversed(topo):
            node._backward()

    def id(self):
        return str(id(self))
