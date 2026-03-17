#### https://SenatorovAI.com

# LSQR Solver — Sparse Linear System Course

[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-live-blue.svg)](https://senatorovai.github.io/LSQR-solver-course/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818738.svg)](https://doi.org/10.5281/zenodo.18821045)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)]()
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)]()

> 🚀 Professional implementation and mathematical explanation of the **LSQR algorithm** for solving large-scale sparse linear systems.

---

## 🔥 Project Overview

This repository provides a complete course-style implementation of the **LSQR algorithm**, including:

- Mathematical derivation
- Connection to Conjugate Gradient
- Sparse matrix optimization
- Numerical stability analysis
- Python implementation from scratch
- Applications in large-scale optimization

---

## Keywords

```

lsqr solver
lsqr algorithm
sparse linear solver
iterative least squares
large scale linear systems
lsqr python implementation
numerical linear algebra
sparse matrix solver
least squares solver
cg based least squares

```

---

## 📚 Mathematical Background

LSQR solves the least squares problem:

$$
\min_x ||Ax - b||_2
$$

Where:

- $$A \in \mathbb{R}^{m \times n}$$ (sparse or large matrix)
- $$b \in \mathbb{R}^m$$
- $$x$$ — unknown vector

---

## 🔵 Algorithm Foundation

LSQR is based on:

- Lanczos bidiagonalization
- Krylov subspace methods
- Connection to Conjugate Gradient on normal equations

Instead of solving:

$$
A^T A x = A^T b
$$

Directly (which is unstable),

LSQR solves it iteratively in a numerically stable way.

---

## ⚡ Core Idea

LSQR constructs Krylov subspaces:

$$
\mathcal{K}_k(A^T A, A^T b)
$$

and minimizes the least squares residual inside this subspace.

It avoids explicitly forming:

$$
A^T A
$$

Which improves:

✅ Stability  
✅ Memory efficiency  
✅ Performance for sparse matrices  

---

## 🧠 Why This Project Is Important

LSQR is used in:

- Large-scale data fitting
- Image reconstruction
- Tomography
- Inverse problems
- Machine learning regularization
- Scientific computing

It is one of the most important industrial-grade solvers.

---

## 🏗 Project Structure

```

lsqr-solver-course/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── src/
│   ├── lsqr_solver.py
│   ├── bidiagonalization.py
│   ├── operators.py
│
├── examples/
│   └── demo.py
│
├── docs/
│   ├── theory.md
│   ├── convergence.md
│
├── images/
│   └── convergence_plot.png
│
└── index.html

````

Clean structure improves:

✔ Discoverability  
✔ Professional appearance  
✔ Research credibility  

---

## 🐍 Example — Simple LSQR Skeleton

```python
import numpy as np

def lsqr(A, b, tol=1e-8, max_iter=1000):
    m, n = A.shape
    x = np.zeros(n)

    r = b - A @ x
    beta = np.linalg.norm(r)
    u = r / beta

    for _ in range(max_iter):
        v = A.T @ u
        alpha = np.linalg.norm(v)
        v = v / alpha

        # Update solution (simplified step)
        x += (beta / alpha) * v

        r = b - A @ x
        beta = np.linalg.norm(r)

        if beta < tol:
            break

    return x
````

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

Run example:

```bash
python examples/demo.py
```

---

## 📊 Visualization (Recommended)

Add:

* Residual norm vs iteration
* Convergence curve
* Krylov subspace dimension growth

Example:

```python
import matplotlib.pyplot as plt

plt.plot(residual_history)
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.title("LSQR Convergence")
plt.show()
```


