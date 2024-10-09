from minigrad.nn.neuron import Neuron


class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]
        self.n_input = n_input
        self.n_output = n_output

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        p = []
        for n in self.neurons:
            p.extend(n.parameters())
        return p
