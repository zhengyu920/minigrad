from minigrad import Value, tanh
import random


class Neuron:
    def __init__(self, n_input):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_input)]
        self.b = Value(0)
        self.n_input = n_input

    def __call__(self, x):
        o = sum([wi * xi for wi, xi in zip(x, self.w)]) + self.b
        return tanh(o)

    def parameters(self):
        return self.w + [self.b]
