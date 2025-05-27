# Solving Coupled Spring-Mass Systems with Neural ODEs: A Modern Approach to a Classical Problem

## Introduction

Have you ever wondered if today’s deep learning tools can solve problems from your high school physics class? What if neural networks could not only approximate solutions to differential equations, but actually learn the dynamics behind them?

In this project, we explore a classic problem from mechanics—the **coupled spring-mass system**—and give it a modern twist using **Neural Ordinary Differential Equations (Neural ODEs)**. Traditionally, such problems are tackled with analytical methods or numerical solvers. Here, we take an ML-inspired path, where the model learns the system dynamics directly from data using the **shooting method** approach described in the [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html).

Why does this matter? Because this hybrid approach bridges classical mathematical modeling with neural network approximations, and opens up new possibilities for data-driven simulations, control systems, and physics-informed machine learning.

## Project Setup

This project was developed in a clean Python environment using `pipenv` for dependency management. To replicate the results or run the code on your own system, follow the steps below.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/neural-ode-spring-mass.git
cd neural-ode-spring-mass
```

### Step 2: Install Dependencies with Pipenv

```bash
pipenv install
pipenv shell
```

### Step 3: Run the Training Script

```bash
python main.py --train
python main.py --train --incremental
python main.py --train --showplot False
python main.py --eval
python main.py --train --solver rk4 --lr 0.001 --segment_size 100
```

### Required Python Libraries

| Library         | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `torch`         | The deep learning framework used to define and train the neural network. |
| `torchdiffeq`   | A PyTorch-compatible library for solving differential equations using Neural ODEs. |
| `numpy`         | For general numerical operations and array management.                  |
| `matplotlib`    | Used to visualize trajectories, loss curves, and model outputs.         |
| `scipy`         | Provides the reference solver and original shooting method implementation for validation. |

## Technical Approach

This project uses **Neural Ordinary Differential Equations (Neural ODEs)** to learn the dynamics of a classical spring-mass system.

### Neural ODEs: A Quick Primer

Neural ODEs represent the change in a system's state as:

```math
\frac{dy}{dt} = f_\theta(y, t)
```

```python
from torchdiffeq import odeint
y_hat = odeint(neural_net_dynamics, y0, t, method='dopri5')
```

### Two-Phase Training Strategy

**Phase 1: Full Trajectory Training**  
Train on the complete trajectory.

**Phase 2: Incremental Window Reduction**  
Retrain on smaller, progressive segments for improved local accuracy.

### Fourier Analysis

```python
x_freq = np.fft.fft(x_raw)
x_filtered = inverse_filter_high_freq(x_freq, threshold=30)
```

This preprocessing improved smoothness and reduced collocation error.

### Integration Method Comparison

| Metric              | `rk4`         | `dopri5`       |
|---------------------|---------------|----------------|
| Final Endpoint MSE  | 0.028         | **0.011**      |
| Avg Collocation Loss| 0.094         | **0.038**      |
| Fit Stability       | Moderate drift| **Stable**     |
| Training Time       | Faster        | Slightly slower|

```python
y_hat = odeint(model, y0, t, method='rk4')
y_hat = odeint(model, y0, t, method='dopri5')
```

## Experimental Results

### 1. Full vs. Incremental Training

- Full training overfit early dynamics
- Incremental improved endpoint accuracy

### 2. Fourier Analysis

- Enhanced fit smoothness
- Reduced endpoint error

### 3. Integration Method

- `dopri5` yielded better endpoint prediction and fit stability

### 4. Uncertainty Estimation

Monte Carlo Dropout offered insight into predictive uncertainty.

## Conclusion

Neural ODEs provide a hybrid model that merges physical systems with machine learning.

### What We Learned

- Two-phase training improves accuracy
- Fourier analysis boosts fit quality
- `dopri5` outperforms `rk4` in long-term stability

### Final Takeaways

Neural ODEs are a viable tool for hybrid modeling—combining physics and learning.

## Get Involved & Explore More

- **GitHub**: [github.com/your-username/neural-ode-spring-mass](https://github.com/your-username/neural-ode-spring-mass)
- **Leave a Comment** and follow for more.
- **Reach Out** via LinkedIn or GitHub.

## References

### Core Concepts & Tools

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).  
  *Neural Ordinary Differential Equations*.  
  [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)

- SciPy Cookbook – Coupled Spring-Mass System:  
  [https://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html](https://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html)

- torchdiffeq GitHub Repository:  
  [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

- PyTorch Documentation:  
  [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

- NumPy Documentation:  
  [https://numpy.org/doc/](https://numpy.org/doc/)

- Matplotlib Documentation:  
  [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

- SciPy Documentation:  
  [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)

### Articles by Shuai Guo, PhD

- *Physics-Informed Neural Networks: An Application-Centric Guide*  
  [https://medium.com/data-science/physics-informed-neural-networks-an-application-centric-guide-dc1013526b02](https://medium.com/data-science/physics-informed-neural-networks-an-application-centric-guide-dc1013526b02)

- *Using Physics-Informed Neural Networks as Surrogate Models: From Promise to Practicality*  
  [https://shuaiguo.medium.com/using-physics-informed-neural-networks-as-surrogate-models-from-promise-to-practicality-3ff13c1320fc](https://shuaiguo.medium.com/using-physics-informed-neural-networks-as-surrogate-models-from-promise-to-practicality-3ff13c1320fc)

- *The Reality of Physics-Informed Neural Networks: Challenges, Alternatives, and Promising Use Cases*  
  [https://medium.com/data-science-collective/the-reality-of-physics-informed-neural-networks-challenges-alternatives-and-promising-use-cases-654d3125785c](https://medium.com/data-science-collective/the-reality-of-physics-informed-neural-networks-challenges-alternatives-and-promising-use-cases-654d3125785c)

- *Modeling Dynamical Systems With Neural ODE: A Hands-on Guide*  
  [https://medium.com/data-science/modeling-dynamical-systems-with-neural-ode-a-hands-on-guide-71c4cfdb84dc](https://medium.com/data-science/modeling-dynamical-systems-with-neural-ode-a-hands-on-guide-71c4cfdb84dc)

- *Using Monte Carlo to Quantify the Model Prediction Error*  
  [https://medium.com/data-science/how-to-quantify-the-prediction-error-made-by-my-model-db4705910173](https://medium.com/data-science/how-to-quantify-the-prediction-error-made-by-my-model-db4705910173)
