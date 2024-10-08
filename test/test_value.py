import minigrad
import torch
import unittest


class TestValue(unittest.TestCase):

    def test_add(self):
        a = minigrad.Value(2.0)
        b = minigrad.Value(3.0)
        c = a + b
        c.backward()

        at = torch.tensor(2, dtype=torch.float32, requires_grad=True)
        bt = torch.tensor(3, dtype=torch.float32, requires_grad=True)
        ct = at + bt
        ct.backward()

        self.assertEqual(c.data, ct.item())
        self.assertEqual(a.grad, at.grad)
        self.assertEqual(b.grad, bt.grad)

    def test_radd(self):
        a = minigrad.Value(2.0)
        b = 3.0
        c = b + a
        c.backward()

        at = torch.tensor(2, dtype=torch.float32, requires_grad=True)
        bt = 3.0
        ct = bt + at
        ct.backward()

        self.assertEqual(c.data, ct.item())
        self.assertEqual(a.grad, at.grad)


if __name__ == '__main__':
    unittest.main()
