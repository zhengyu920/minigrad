from minigrad.nn.layer import Layer


class MLP:
    def __init__(self, n_input, layer_sizes):
        n_in = n_input
        self.layers = []
        for ls in layer_sizes:
            self.layers.append(Layer(n_in, ls))
            n_in = ls

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        p = []
        for l in self.layers:
            p.extend(l.parameters())
        return p
