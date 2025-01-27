""" Here is an example of a system of first order ODEs solved using PINN.

The example was derived from this YouTube `[1]`_ which was developed using 
Tensorflow.  I rewrote, with the help of ChatGPT 4o and Llama 3.2 the code into
PyTorch.

The system of equations (ODEs):

    dx/dt = -2x - y = 0 with Initial Condition (IC) x(0) = 1 for t ∈ [0, 5]
    dy/dt = -x - 2y = 0 with Initial Condition (IC) y(0) = 0

The analytical solution is:

    x(t) = 1/2 exp(-t) + 1/2 exp(-3t)
    y(t) = -1/2 exp(-t) + 1/2 exp(-3t)

Tensorflow code that I trascribed from `[1]`_:

import tensorflow as tf

class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activiation='tanh')
        self.dense2 = tf.keras.layers.Dense(64, activiation='tanh')
        self.dense3 = tf.keras.layers.Dense(2, activiation=None) # x(t), y(t)

    def call(self, t):
        t = self.dense1(t)
        t = self.dense2(t)
        return self.dense3(t)

def loss_fn(model, t):
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        tape1.watch(t)
        tape2.watch(t)
        output = model(t)
        x = output[:, 0:1]
        y = output[:, 1:2]
        dxdt = tape1.gradient(x, t)
        dydt = tape2.gradient(y, t)

        # Residual ODEs
        res_x = dxdt + 2x + y
        res_y = dydt + x + 2y

        # IC
        ICx = tf.square(x[0] - 1)
        ICy = tf.square(y[0] - 0)

        # Total loss
        loss = tf.reduce_mean(tf.square(res_x)) +
               tf.reduce_mean(tf.square(res_y)) +
               ICx + ICy
        
        return loss

def train(model, t, epochs, optimizer):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, t)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply.gradients(zip(grads, model.training_variables))
        if epoch % 500 == 0:
            print(f"Epoch {epoch} Loss: {loss.numpy()}")

.. _[1]: https://www.youtube.com/watch?v=gXv1SGoL04c

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.dense3 = nn.Linear(64, 2)  # Outputs x(t), y(t)

    def forward(self, t):
        t = self.dense1(t)
        t = self.dense2(t)
        return self.dense3(t)
    
def loss_fn(model, t):
    """ Total Loss Function """
    # Enable gradient tracking
    t.requires_grad_(True)

    # Forward pass
    output = model(t)
    x = output[:, 0:1]
    y = output[:, 1:2]

    # Compute gradients dx/dt and dy/dt
    dxdt = torch.autograd.grad(outputs=x, inputs=t,
                                grad_outputs=torch.ones_like(x),
                                create_graph=True)[0]
    dydt = torch.autograd.grad(outputs=y, inputs=t,
                                grad_outputs=torch.ones_like(y),
                                create_graph=True)[0]

    # Residual ODEs
    res_x = dxdt + 2 * x + y
    res_y = dydt + x + 2 * y

    # Initial Conditions (IC)
    ICx = (x[0] - 1) ** 2
    ICy = (y[0] - 0) ** 2

    # Total loss
    loss = (torch.mean(res_x ** 2) +
            torch.mean(res_y ** 2) +
            ICx + ICy)

    return loss

def train(model, t, epochs, optimizer):
    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute the loss
        loss = loss_fn(model, t)
        
        # Backpropagation
        loss.backward()
        
        # Update the model parameters
        optimizer.step()
        
        # Print loss periodically
        if epoch % 500 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")

# Run our model
model = PINN()

# Define the optimizer
# optimizer = tf.keras.optimizer.Adam(learning_rate=0.001) # control rate
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Control rate

# Define domain (t ∈ [0, 5])
# t = tf.convert_to_tensor(np.linspace(0,5,100)[:, None], dtype=tf.float32)
t = torch.tensor(np.linspace(0, 5, 100).reshape(-1, 1), dtype=torch.float32)

# Train the model
train(model, t, epochs=4000, optimizer=optimizer)

# Test the model
t_test = torch.tensor(np.linspace(0, 5, 300).reshape(-1, 1), dtype=torch.float32)
with torch.no_grad():
    predictions = model(t_test)
    x_pred, y_pred = predictions[:, 0].numpy(), predictions[:, 1].numpy()

# Test the model
# t_test = tf.convert_to_tensor(np.linspace(0,5,300)[:, None], dtype=tf.float32)
# x_pred, y_pred = model(t_test).numpy().T
t_test = torch.tensor(np.linspace(0, 5, 300).reshape(-1, 1), dtype=torch.float32)
with torch.no_grad():
    predictions = model(t_test)
    x_pred, y_pred = predictions[:, 0].numpy(), predictions[:, 1].numpy()

# Truth solution
def solution(t):
    t = np.asarray(t)  # Ensure it's a standard ndarray
    x_t = 0.5 * (np.exp(-t) + np.exp(-3*t))
    y_t = 0.5 * (-1*np.exp(-t) + np.exp(-3*t))
    return x_t, y_t

# Compare results:
x_true, y_true = solution(t_test)

plt.close('all')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t_test, x_true, label=r'Analytical $x(t)$', color='red')
plt.plot(t_test, x_pred, label=r'PINN $x(t)$', color='blue')
plt.title(r'PINNs vs Analytical Solution $x(t)$', fontsize=14)
plt.xlabel(r'Time $t$')
plt.ylabel(r'$x(t)$')
plt.grid(True)
plt.legend(fontsize=12, loc="upper right")

plt.subplot(1,2,2)
plt.plot(t_test, y_true, label=r'Analytical $y(t)$', color='red')
plt.plot(t_test, y_pred, label=r'PINN $y(t)$', color='blue')
plt.title(r'PINNs vs Analytical Solution $y(t)$', fontsize=14)
plt.xlabel(r'Time $t$')
plt.ylabel(r'$y(t)$')
plt.grid(True)
plt.legend(fontsize=12, loc="upper right")

plt.tight_layout()
plt.show()

def generate_noisy_data(t_values, noise_scale=0.05, seed=None):
    """
    Generates noisy data for the solution of two first-order ODEs.

    Parameters:
    - t_values (array-like): Time values for the solution.
    - noise_scale (float): Standard deviation of the Gaussian noise (default: 0.05).
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - t_values (ndarray): Input time values.
    - x_noisy (ndarray): Noisy data for x_t.
    - y_noisy (ndarray): Noisy data for y_t.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get the exact solution
    x_exact, y_exact = solution(t_values)
    
    # Add Gaussian noise
    x_noisy = x_exact + np.random.normal(scale=noise_scale, size=x_exact.shape)
    y_noisy = y_exact + np.random.normal(scale=noise_scale, size=y_exact.shape)
    
    return t_values, x_noisy, y_noisy

# Example usage:
t_values = np.linspace(0, 5, 50)  # Generate time values
t, x_noisy, y_noisy = generate_noisy_data(t_values, noise_scale=0.1, seed=42)

# Print a sample of the noisy data
for i in range(5):
    print(f"t={t[i]:.2f}, x_noisy={x_noisy[i]:.4f}, y_noisy={y_noisy[i]:.4f}")