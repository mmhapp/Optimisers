# Optimisers from Scratch

Gradient descent, Momentum and Adam implemented from scratch with pure NumPy. All three are run on the same poorly conditioned loss surface so the differences are easy to see.

![Paths](paths.png)

## How this relates to real AI models

Every neural network — from a small CNN classifying images to a multi-billion-parameter language model — is trained by repeatedly taking optimiser steps.  
Adam (and its variants AdamW, Lion, …) is the default optimiser in virtually all modern training runs of transformers, diffusion models and vision networks. Understanding why plain gradient descent struggles on ill-conditioned surfaces, and how momentum / adaptive scaling help, is foundational to training any deep model successfully.

## Running

```bash
pip install -r requirements.txt
python main.py        # prints final losses, saves paths.png and loss_curves.png
```

## Mathematics

Training is an optimisation problem: find parameters $\theta$ that minimise a loss $L(\theta)$.

### 1. Gradient descent

The gradient $\nabla_\theta L$ points in the direction of steepest ascent, so we step in the opposite direction:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta)$$

where $\eta$ is the learning rate.

**Why choosing $\eta$ is hard.** Consider the quadratic surface $L = \tfrac12(a x^2 + b y^2)$ with $a \ll b$. After one GD step the coordinates are multiplied by $(1-\eta a)$ and $(1-\eta b)$. For the steep direction not to explode we need $|1-\eta b| < 1$, i.e.

$$\eta < \frac{2}{b}.$$

But then in the flat direction the factor is $1-\eta a \approx 1$ and progress is slow. The condition number $\kappa = b/a$ quantifies how bad the situation is. In this demo $\kappa = 100$.

### 2. Momentum

We accumulate a velocity that remembers previous gradients:

$$u_t = \beta \, u_{t-1} + g_t, \qquad \theta \leftarrow \theta - \eta \, u_t$$

where $g_t = \nabla_\theta L(\theta_{t-1})$. In the steep direction the gradient sign oscillates and the sum cancels; in the flat direction gradients point the same way and accumulate. For a constant gradient the effective step size becomes

$$\eta \, \sum_{k\ge 0}\beta^k = \frac{\eta}{1-\beta},$$

so $\beta = 0.9$ gives roughly a 10× larger step in the flat direction.

### 3. Adam

Adam keeps running estimates of both the first moment (mean) and the second moment (uncentred variance) of the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2$$

Because $m_0 = v_0 = 0$ the estimates are biased toward zero at the beginning. Bias correction:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

Update:

$$\theta \leftarrow \theta - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

Dividing by $\sqrt{\hat v_t}$ normalises the step size of every parameter individually: parameters that receive large gradients move cautiously, those with small gradients move more boldly. This is why Adam works well without having to hand-tune a learning rate for every direction.

## Gradient checking

`main.py` compares the analytic gradient with a numerical central difference:

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}$$

This is a useful way to catch gradient bugs before training starts.

## What the experiment shows

| Method | Behaviour |
|---|---|
| Gradient descent | Stable, but progress in the flat direction is slow |
| Momentum | Faster, but overshoots the optimum and oscillates |
| Adam | Fast because step size adapts per parameter; slight overshoot then quick settling |

Results depend on the chosen learning rates. Try changing the values in `main.py` and the condition number $A,B$.

## References

- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*
- Goh (2017), *Why Momentum Really Works*, Distill
- Dive into Deep Learning, Chapter 12 (Optimization Algorithms)