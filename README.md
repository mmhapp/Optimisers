# Optimisers

Gradient descent, Momentum and Adam implemented from scratch with NumPy and run on the same loss surface. The picture below shows how the optimisers move towards the minimum of the loss function $L(\theta)$, where $\theta=(\theta_0,\theta_1)$. The contour lines connect points with the same loss, every optimiser starts from the same point, and the star marks the optimum. Gradient descent is stable, but slow in the flat direction. Momentum is faster, but overshoots and oscillates. Adam adapts the step size per parameter, with slight overshoot and quick settling.

<img src="paths.png" alt="Paths" width="100%">
<sub>Figure 1: Optimointimenetelmien häviökäyrät (log-asteikko).</sub>

---

## Relation to Artificial Intelligence

Every neural network – from a small convolutional neural network (CNN) to a large language model (LLM) – is trained by repeatedly taking optimiser steps. Adam is the default optimiser in virtually all modern training of transformers, diffusion models and vision networks. Understanding why plain gradient descent struggles on ill-conditioned surfaces, and how momentum and adaptive scaling help, is foundational to training deep learning (DL) models. When one calls `optimizer.step()` in PyTorch, the framework applies the update rules `torch.optim.SGD` and `torch.optim.Adam`, using the gradients that backpropagation has just computed.

## Mathematics

Training is an optimisation problem: find parameters $\theta=(\theta_0,\theta_1)$ that minimise a loss $L(\theta)$.

<ins>1. Gradient descent</ins>

The gradient $\nabla_\theta L$ points in the direction of steepest ascent, so we step in the opposite direction as

$\theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta)$,

where $\eta$ is the learning rate.

Why choosing $\eta$ is hard? Consider the quadratic surface $L = \tfrac12(a x^2 + b y^2)$ with $a \ll b$. After one GD step the coordinates are multiplied by $(1-\eta a)$ and $(1-\eta b)$. For the steep direction not to explode we need $\lvert 1-\eta b\rvert < 1$, i.e.

$\eta < \frac{2}{b}$.

But then in the flat direction the factor is $1-\eta a \approx 1$ and progress is slow. The condition number $\kappa = b/a$ quantifies how bad the situation is. In this demo $\kappa = 100$.

<ins>2. Momentum</ins>

We accumulate a velocity that remembers previous gradients as

$u_t = \beta\,u_{t-1} + g_t$,

and update the parameters as

$\theta \leftarrow \theta - \eta\,u_t$,

where $g_t = \nabla_\theta L(\theta_{t-1})$. In the steep direction the gradient sign oscillates and the sum cancels; in the flat direction gradients point the same way and accumulate. For a constant gradient the effective step size becomes

$\eta\,\sum_{k\ge 0}\beta^k = \frac{\eta}{1-\beta}$,

so $\beta = 0.9$ gives roughly a 10× larger step in the flat direction.

<ins>3. Adam</ins>

Adam keeps running estimates of both the first moment (mean) and the second moment (uncentred variance) of the gradient as

$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$.

Because $m_0 = v_0 = 0$ the estimates are biased toward zero at the beginning. Bias correction is

$\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$.

The update is

$\theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$.

Dividing by $\sqrt{\hat v_t}$ normalises the step size of every parameter individually: parameters that receive large gradients move cautiously, those with small gradients move more boldly. This is why Adam works well without having to hand-tune a learning rate for every direction.

## Running

```bash
pip install -r requirements.txt
python main.py
```

## Project

```
├── data/               # Benchmark datasets
├── Results/            # Output directory for weights and predictions
├── fm.py               # Main execution script
├── fmloop.py           # Automated experiment runner
├── settings.py         # Hyperparameter configurations
└── statistics.pdf      # Visualisation and analysis
```

## References

Zhang, A., Lipton, Z. C., Li, M. & Smola, A. J. (2023). “Optimization Algorithms” [Chapter 9]. Dive into Deep Learning. Available at: https://d2l.ai/chapter_optimization/