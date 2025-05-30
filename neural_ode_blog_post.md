# Solving Coupled Spring-Mass Systems with Neural ODEs: A Modern Approach to a Classical Problem

## Introduction

Have you ever wondered if today’s deep learning tools can solve problems from your high school physics class? What if neural networks could not only approximate solutions to differential equations, but actually learn the dynamics behind them?

In this project, we explore a classic problem from mechanics—the **coupled spring-mass system**—and give it a modern twist using **Neural Ordinary Differential Equations (Neural ODEs)**. Traditionally, such problems are tackled using analytical methods or numerical solvers, including finite difference methods, Runge-Kutta integration schemes, and the shooting method [Ascher & Petzold, 1998; Butcher, 2008; Press et al., 2007]. Here, we take an ML-inspired path, where the model learns the system dynamics directly from data using the **shooting method** approach described in the [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html).

Why does this matter? Because this hybrid approach bridges classical mathematical modeling with neural network approximations, and opens up new possibilities for data-driven simulations, control systems, and physics-informed machine learning. In many experimental or laboratory settings, collecting dense data over an entire time series is impractical or expensive. With Neural ODEs, we can often learn accurate system dynamics from sparse or partial observations—something that traditional numerical methods struggle with unless provided with highly resolved datasets. This makes Neural ODEs especially valuable in scenarios where measurement is limited but insight into system behavior is still essential.

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

The training script supports a command-line interface that allows you to override default argument values using flags defined in the `parse_args` function.

To run the script with default settings:

```bash
python neural_sysEq_parameters_ft_reduxtime.py
```

To customize the training, override one or more arguments directly in the command line. Examples:

- **Change the dropout rate**:
  ```bash
  python neural_sysEq_parameters_ft_reduxtime.py --dropout 0.2
  ```

- **Set the number of Monte Carlo samples**:
  ```bash
  python neural_sysEq_parameters_ft_reduxtime.py --mc_samples 50
  ```

- **Switch the solver to RK4**:
  ```bash
  python neural_sysEq_parameters_ft_reduxtime.py --solver rk4
  ```

- **Run with incremental training**:
  ```bash
  python neural_sysEq_parameters_ft_reduxtime.py --incremental
  ```

- **Disable plotting during training**:
  ```bash
  python neural_sysEq_parameters_ft_reduxtime.py --showplot False
  ```

Use `--help` to view all available options:

```bash
python neural_sysEq_parameters_ft_reduxtime.py --help
```

#### Available Command-Line Arguments
This flexible interface lets you experiment with different configurations without modifying the source code. Below is a list of all available arguments as defined in the script’s `parse_args` function:

| Argument                 | Type      | Default                                | Description |
|--------------------------|-----------|----------------------------------------|-------------|
| `--device`               | str       | `'cpu'`                                | Device to run the model on (`cpu` or `cuda`). |
| `--hidden_size`          | int list  | `[256,128,256,128,256,128]`            | List of hidden layer sizes for the NN. |
| `--activation_fn`        | str       | `'SIREN'`                              | Activation function: `SiLU`, `Tanh`, or `SIREN`. |
| `--dropout_rate`         | float     | `0.4`                                  | Dropout rate applied between hidden layers. |
| `--integrator`           | str       | `'dopri5'`                             | ODE solver used for training: `rk4` or `dopri5`. |
| `--test_integrator`      | str       | `'rk4'`                                | ODE solver used for evaluation. |
| `--rtol`                 | float     | `1e-3`                                 | Relative tolerance for `dopri5`. |
| `--atol`                 | float     | `1e-4`                                 | Absolute tolerance for `dopri5`. |
| `--lambda_cont`          | float     | `0.1`                                  | Continuity loss weight between segments. |
| `--lambda_ic`            | float     | `0.01`                                 | Initial condition loss weight. |
| `--lambda_colloc`        | float     | `1.0`                                  | Collocation loss weight at segment midpoints. |
| `--lambda_end`           | float     | `0.01`                                 | Endpoint loss weight. |
| `--lambda_spec_mag`      | float     | `0.001`                                | Spectral magnitude loss weight for x1. |
| `--lambda_spec_phase`    | float     | `0.001`                                | Spectral phase loss weight for x1. |
| `--lambda_spec_mag2`     | float     | `0.01`                                 | Spectral magnitude loss weight for x2. |
| `--lambda_spec_phase2`   | float     | `0.001`                                | Spectral phase loss weight for x2. |
| `--lambda_energy`        | float     | `0.001`                                | Energy dissipation penalty weight. |
| `--num_epochs`           | int       | `100`                                  | Total number of training epochs. |
| `--print_every`          | int       | `10`                                   | Print loss values every N epochs. |
| `--t0`                   | float     | `0.0`                                  | Start time of simulation. |
| `--tN`                   | float     | `10.0`                                 | End time of simulation. |
| `--delta_t`              | float     | `1.0`                                  | Time reduction step for incremental training. |
| `--mc_runs`              | int       | `50`                                   | Number of Monte Carlo dropout runs. |
| `--no-showplot`          | flag      | `True`                                 | Use `--no-showplot` to disable plot display. |
| `--K`                    | int       | `20`                                   | Number of time segments. |
| `--batch_size`           | int       | `1`                                    | Training batch size. |
| `--state_dim`            | int       | `4`                                    | Number of state variables. |
| `--p`                    | float list| `[1.0,1.5,8.0,40.0,0.5,1.0,0.8,0.5]`    | Physics parameters: `[m1,m2,k1,k2,L1,L2,b1,b2]`. |
| `--init_cond`            | float list| `[0.5,0.0,2.25,0.0]`                   | Initial state values: `[x1,y1,x2,y2]`. |

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

### Fourier Analysis

```python
x_freq = np.fft.fft(x_raw)
x_filtered = inverse_filter_high_freq(x_freq, threshold=30)
```

This preprocessing improved smoothness and reduced collocation error.

### Constraining the training with losses
During training, we enforce alignment with the real data by fitting at **collocation points**—key sample points collected from experimental or simulated trajectories. Loss functions are computed at these points and may include terms for:

- Collocation accuracy (Neural ODE match to ground truth)
- Continuity between time segments
- Initial condition agreement
- Endpoint convergence
- Frequency-domain behavior (via spectral loss)


### Two-Phase Training Strategy

**Phase 1: Full Trajectory Training**  
Train on the complete trajectory. This is the standard approach for training a Neural ODE over an entire time domain. The goal is to learn the system's dynamics across the full trajectory, from initial to final time, using both known physical equations and a neural network that accounts for unknown or complex behaviors not easily modeled analytically.

The outcome of this phase is a model that generally captures the full dynamic behavior but may require refinement for fine-grained prediction or endpoint accuracy—leading us to Phase 2.

**Phase 2: Incremental Window Reduction**  
Retrain on smaller, progressive segments for improved local accuracy. In this setup, we combine a physics-based ODE (e.g., Newtonian motion) with a trainable neural network that captures residual dynamics. This hybrid formulation allows the system to stay grounded in physical law while offering flexibility to model missing or uncertain components.



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
