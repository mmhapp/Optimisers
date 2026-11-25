"""Gradient descent, Momentum and Adam from scratch (NumPy only).

Each optimiser is a small class with a method step(theta, grad) -> new theta.
"""
import numpy as np


class GradientDescent:
    """theta <- theta - lr * grad"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, theta, grad):
        return theta - self.lr * grad


class Momentum:
    """u_t = beta * u_{t-1} + g_t ;  theta <- theta - lr * u_t"""

    def __init__(self, lr=0.01, beta=0.9):
        self.lr, self.beta = lr, beta
        self.u = None

    def step(self, theta, grad):
        if self.u is None:
            self.u = np.zeros_like(theta)
        self.u = self.beta * self.u + grad
        return theta - self.lr * self.u


class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)  # bias correction
        v_hat = self.v / (1 - self.beta2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
