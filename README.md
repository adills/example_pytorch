# Examples of using PyTorch

Author: @adills

## Setup
I use `pipenv` for environment setup as defined the the Pipfile.

## Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs), and their various implementations, have gained significant traction over the past decade across science and engineering disciplines. Their popularity stems from their unique ability to seamlessly integrate neural network architectures with fundamental physical laws.

PINNs excel at modeling differential equations—both ordinary (ODEs) and partial (PDEs)—including complex systems of these equations. The core innovation lies in their loss function design, which incorporates both:

1. **Data-driven components**: Training on available measurement data
2. **Physics-driven components**: Enforcing governing equations as soft constraints

This dual-objective optimization allows PINNs to discover solutions that simultaneously fit observed data while respecting underlying physical principles. The network learns continuous functional representations that automatically satisfy conservation laws, boundary conditions, and initial conditions without requiring traditional numerical discretization schemes.

PINNs have demonstrated remarkable effectiveness in fluid dynamics, heat transfer, quantum mechanics, and other domains where complex physical phenomena require efficient and accurate computational models.

## Neural ODEs with the Shooting Method

Neural Ordinary Differential Equations (Neural ODEs) represent a continuous-depth approach to deep learning where layer transformations are defined by an ODE. The shooting method is an optimization technique used in combination with Neural ODEs to improve training efficiency and stability.

In a Neural ODE with shooting method:

1. **Core concept**: Instead of directly integrating through the entire ODE trajectory, the shooting method divides the integration interval into segments with intermediate "checkpoints."

2. **Forward pass**: The neural network predicts initial conditions at each checkpoint, which are then integrated forward. This creates shorter integration paths that are more stable.

3. **Backward pass**: Gradients are computed locally within segments rather than through the entire trajectory, reducing the risk of vanishing/exploding gradients.

4. **Optimization**: The method optimizes both the ODE parameters and the checkpoint states, allowing for parallel computation and memory efficiency.

This approach effectively "shoots" from predicted intermediate states toward target states, iteratively refining the trajectory through the continuous neural network, resulting in more stable training for long time-horizon problems.
my examples are inspired from various examples that I've found in academic literature, GitHub repositories, and Medium articles.  my documentation is limited but I try to property reference the materials I used within the code itself. 

## Physics-Informed Neural Networks (PINNs) with the Shooting Method

Physics-Informed Neural Networks (PINNs) combine neural networks with physical laws encoded as differential equations. The shooting method enhances PINNs by introducing an efficient solution strategy for complex physical systems.

In a PINN with shooting method:

1. **Core concept**: The shooting method divides the computational domain into multiple segments with intermediate checkpoints, allowing for localized integration within each segment.

2. **Physics enforcement**: The neural network still minimizes a combined loss that includes both data fitting and physics-based constraints, but does so over shorter trajectory segments.

3. **Boundary conditions**: Initial conditions at each checkpoint are predicted by the network and optimized alongside model parameters, creating a piecewise approximate solution that satisfies the governing equations.

4. **Computational advantage**: This approach reduces integration path lengths, improving numerical stability while maintaining physical consistency across the entire domain.

5. **Convergence**: The method iteratively refines both the physical trajectory and checkpoint states, effectively "shooting" from intermediate predictions toward physically consistent solutions.

This modification enhances traditional PINNs by making them more tractable for stiff or long time-horizon physical systems while preserving their physics-informed learning capabilities.

## Examples
I have two examples using a system of linear ordinary differential (ODE) equations similar to a couples spring mass system: 
1. I started with a basic PINN that determines the displacements that over the ODEs (`sysEqns.ipynb` and `sysEqns.py`).  
2. Next, I examined the performance of a Neural ODE in which the shooting method is used to predict the ending point (`neural_sysEq.ipynb` and `neural_sysEq.py`).
3. The `neural_sysEq_parameters.py` incorporates a more general coupled spring-mass system
that allows the user to define the masses, spring coefficients, and dampening terms.

## Neural ODEs with Fourier spectral loss constraints. 
The more general coupled spring-mass system as in `neural_sysEq_parameters.py` proved
to be very difficult to solve using the traditional Neural ODE approach.  So I explored
additional Fourier methods in `neural_sysEq_parameters_ft.py`.  Here is a summary of the
changes I made:

Here’s a high-level changelog of what I tried—each tweak, why I did it, and its net impact:
1.	Modular refactor
- Introduced parse_args(), create_data(), a Trainer class, and standalone plot functions
- Effect: Clean CLI, easy hyperparameter control, and clear separation of data prep, training, and visualization.
2.	Unified physics‐only solve
- In create_data(), solve the ODE once over t_grid∪t_test and slice out both sets
- Effect: Guarantees true states at segment knots exactly match the test trajectory—no more interpolation drift.
3.	Multi‐shooting architecture
- Trainable segment initial states s[k], plus losses:
	- 	fit_loss (match each segment’s end to true state)
	- 	cont_loss (continuity between segments)
	- 	colloc_loss (ODE‐residual at midpoints)
	- 	end_loss (track final endpoint error)
- Effect: Robust “divide‐and‐conquer” integration that can capture complex trajectories segment‐by‐segment.
4.	Denser collocation
- Switched from a single midpoint to 5 interior collocation points per segment
- Boosted λ_colloc → 1.0
- Effect: Stronger enforcement of the physics ODE in between boundary points, reducing mid‐segment drift.
5.	Energy‐dissipation regularizer
- Added compute_energy(u) and penalized any increase in mechanical energy over time
- Controlled by λ_energy
- Effect: Physically anchors the network to the true damped behavior—suppresses non-physical overshoots.
6.	Fourier‐feature embedding
- FFT of the true x₁(t) to auto-select dominant frequencies
- Augmented MLP input with sinusoids at those modes
- Effect: Gave the network an explicit basis for high-frequency ripples, boosting its ability to fit fine oscillations.
7.	Spectral‐loss curriculum (and log-magnitude)
- Introduced magnitude & phase FFT‐losses on x₁ and x₂, ramped in over the first 50% of epochs
- Switched to log-magnitude to cap dynamic range
- Effect: Initially improved frequency alignment but tended to destabilize training when fully on—so we later dialed it back.
8.	Learnable Fourier phase & amplitude
- Replaced fixed sin/cos embeddings with trainable φᵢ and Aᵢ per mode
- Reduced input size accordingly (only sine needed)
- Effect: Allowed direct, low-dimensional correction of phase/amplitude offsets—faster convergence on spectral alignment.
9.	Adaptive optimizer & scheduler
- Separate AdamW groups:
    - φ/A at lr = 1e-2, no decay
    - rest + segment states at lr = 1e-3, weight decay
- CosineAnnealingLR over all epochs (ηₘᵢₙ = 1e-5)
- Effect: Stabilized training by cooling the LR continuously, especially as spectral losses kicked in.
10.	Gradient clipping
- Clipped all grads to max‐norm 1.0 after backward()
- Effect: Prevented runaway updates when some loss terms spiked.
11.	SIREN architecture
- Added a Sine activation and let --activation_fn=SIREN replace SiLU/Tanh
- Effect: Provided an inherent oscillatory basis throughout the network—further improved capture of both envelope and ripples.
12.	Light‐touch phase tuning
- Lowered λ_spec_phase and λ_spec_phase2 to 1e-3
- After half the epochs, froze φ/A (requires_grad=False) and early-stopped if phase loss rose
- Effect: Kept the model from “over-chasing” tiny phase errors once the bulk of the dynamics were learned.

⸻

Net outcome:
-   Envelope & ripples are now captured with sub-percent endpoint errors.
- 	Physical damping is enforced via energy regularization.
- 	High-frequency details are modeled through SIREN + Fourier features.
- 	Training stability is maintained by curriculum, adaptive LR, and gradient clipping.

Together they demonstrate a systematic progression from a vanilla PINN to a finely‐tuned, physics‐aligned Neural ODE.