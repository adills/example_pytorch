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