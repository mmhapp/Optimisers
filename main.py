"""Comparison: GD vs Momentum vs Adam on a poorly conditioned loss surface.

Loss surface: L(x, y) = 0.5 * (a * x^2 + b * y^2),  a << b.
The surface is a narrow valley: steep in y, flat in x.
Run:  python main.py   ->  saves paths.png and loss_curves.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optimizers import GradientDescent, Momentum, Adam


A, B = 0.05, 5.0  # Hessian eigenvalues -> condition number B/A = 100


def loss(theta):
    x, y = theta
    return 0.5 * (A * x**2 + B * y**2)


def grad(theta):
    x, y = theta
    return np.array([A * x, B * y])


def numerical_grad(f, theta, h=1e-6):
    """Central difference: verifies the analytic gradient."""
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        g[i] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g


def run(opt, theta0, steps=100):
    theta = theta0.copy()
    path, losses = [theta.copy()], [loss(theta)]
    for _ in range(steps):
        theta = opt.step(theta, grad(theta))
        path.append(theta.copy())
        losses.append(loss(theta))
    return np.array(path), np.array(losses)


def main():
    theta0 = np.array([-4.0, 1.5])
    assert np.allclose(grad(theta0), numerical_grad(loss, theta0), atol=1e-5)

    optimizers = {
        "Gradient descent": GradientDescent(lr=0.35),  # near stability limit 2/B = 0.4
        "Momentum": Momentum(lr=0.1, beta=0.9),
        "Adam": Adam(lr=0.3),
    }
    colors = {"Gradient descent": "#e5534b", "Momentum": "#e3b341", "Adam": "#39c5bb"}
    results = {name: run(opt, theta0) for name, opt in optimizers.items()}

    # --- paths on the loss surface ---
    xs, ys = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-2, 2, 300))
    zs = 0.5 * (A * xs**2 + B * ys**2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.contour(xs, ys, zs, levels=25, cmap="Greys", alpha=0.6)
    for name, (path, _) in results.items():
        ax.plot(path[:, 0], path[:, 1], "-", ms=3, lw=1.5, label=name, color=colors[name])
    ax.plot(0, 0, "*", ms=14, color="black", label="Global minimum")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_title("Optimiser paths on the same loss surface")
    ax.legend()
    fig.tight_layout()
    fig.savefig("paths.png", dpi=500, bbox_inches="tight", pad_inches=0.01)

    # --- loss curves ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (_, losses) in results.items():
        ax.semilogy(losses + 1e-12, label=name, color=colors[name])
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("loss_curves.png", dpi=500)

    for name, (_, losses) in results.items():
        print(f"{name:18s} final loss = {losses[-1]:.3e}")


if __name__ == "__main__":
    main()