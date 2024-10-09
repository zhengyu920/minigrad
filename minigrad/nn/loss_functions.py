def mse_loss(y_hat, y):
    return sum([(yh - ys) ** 2 for yh, ys in zip(y_hat, y)]) / len(y_hat)
