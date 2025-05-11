# %% [markdown]
# # Neural ODE with Multi-Shooting for Coupled Spring–Mass System
#
# This script demonstrates a **physics-informed Neural ODE** that integrates known equations of motion for a coupled spring–mass system alongside a small neural network correction. Key features and workflow:
#
# 1. **Multi-Shooting Segmentation**  
#    - Split the interval \([t_0, t_N]\) into \(K\) segments.  
#    - Each segment has its own trainable initial state, warm-started with the analytic solution.  
#
# 2. **Physics + Neural Correction**  
#    - Define `eom(y)` for the known system dynamics.  
#    - Wrap a `MODEL` network to learn residual forces.  
#    - Combine them in `ODEFunc.forward(t, y)` and integrate via `torchdiffeq.odeint`.
#
# 3. **Composite Loss Function**  
#    - **Physics**: enforce the neural ODE to learn the system ODE at the segment mid-points
#    - **Data-Fit**: match predicted segment-end states to observed data at boundaries.  
#    - **Continuity**: enforce smooth transitions between segments.  
#    - **Initial-Condition Penalty**: anchor the first segment to the true start state.  
#    - **Collocation**: penalize ODE residual at each segment midpoint.  
#
# 4. **Training Setup**  
#    - Optimizer: AdamW with weight decay.  
#    - LR Scheduler: Cosine annealing over `num_epochs`.  
#    - Solver: fixed-step RK4 for efficiency (or adaptive Dormand–Prince).  
#
# 5. **Visualization**  
#    - **Stacked loss plot**: breakdown of all loss components plus total.  
#    - **Endpoint convergence**: predicted vs. true final-state over epochs.  
#    - **Trajectory comparison**: true vs. predicted \(x(t)\) & \(y(t)\), with segment-boundary markers.
#
# **Usage:**  
# - Adjust `K`, epoch count, and loss weights (`λ_cont`, `λ_ic`, `λ_colloc`) to suit your data.  
# - Run the script as-is to train and visualize results.  
# - Replace the analytic `solution(t)` with real observations at segment times to fit measured data.

# %% [markdown]
# ## Set up libaries

# %%
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train physics-informed Neural ODE on coupled spring-mass system")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128,64,128,64,128,64])
    parser.add_argument("--activation_fn", type=str, choices=["SiLU", "Tanh"], default="SiLU",
                        help="Activation function: SiLU or Tanh")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--integrator", type=str, choices=["rk4", "dopri5"], default="rk4")
    parser.add_argument("--test_integrator", type=str, choices=["rk4", "dopri5"], default="rk4")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for dopri5")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for dopri5")
    parser.add_argument("--lambda_cont", type=float, default=1e-1, help="segment continuity loss weight")
    parser.add_argument("--lambda_ic", type=float, default=1e-2, help="initial condition loss weight")
    parser.add_argument("--lambda_colloc", type=float, default=1e-2, help="midpoint collocation loss weight")
    parser.add_argument("--lambda_end", type=float, default=1e-2, help="Endpoint loss weight")
    parser.add_argument("--num_epochs", type=int, default=400, help="Number of epochs to run")
    parser.add_argument("--print_every", type=int, default=50, help="Print epoch loss values at these intervales")
    parser.add_argument("--t0", type=float, default=0.0, help="Start time")
    parser.add_argument("--tN", type=float, default=5.0, help="End time")
    parser.add_argument("--K", type=int, default=20, help="Number of segments")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--state_dim", type=int, default=4, help="State dimension")
    parser.add_argument("--p", type=float, nargs=8, default=[1.0,1.5,8.0,40.0,0.5,1.0,0.8,0.5], help="Physics parameters [m1,m2,k1,k2,L1,L2,b1,b2]")
    parser.add_argument("--init_cond", type=float, nargs=4, default=[0.5,0.0,2.25,0.0], help="Initial conditions [x1,y1,x2,y2]")
    return parser.parse_args()
# %%
device = 'cpu'

# For model:
hidden_size = [128,64,128,64,128,64] # [64,  64]
activation_fn = torch.nn.SiLU # torch.nn.Tanh
dropout_rate = 0
# odefunc implements f_phys + f_NN
integrator = 'rk4'#'dopri5' #'rk4'
test_integrator = 'rk4'
int_options = {'rtol':1e-3, 'atol':1e-4} # for dopri5 integrator

# For loss:
λ_cont = 1e-1 # 1.0
λ_ic = 1e-2
λ_colloc = 1e-2  # weight for midpoint collocation loss
# Endpoint loss weight and tracker
λ_end = 1e-2        # weight for overall endpoint loss

num_epochs = 400
print_every = 50

# %% [markdown]
# ## Definitions

# %%
def solution(t, args):
    """
    Compute numeric solution of the true physics-only system.
    u = [x1, y1, x2, y2]; p and init_cond from args.
    Returns x1, y1, x2, y2 tensors.
    """
    with torch.no_grad():
        # Initial state
        w0 = torch.tensor(args.init_cond, dtype=torch.float32, device=args.device)
        # Physics-only derivative function
        def phys_fun(ti, u):
            dx1, dy1, dx2, dy2 = eom(u, args.p)
            return torch.stack([dx1, dy1, dx2, dy2], dim=-1)
        # Solve ODE
        wsol = odeint(phys_fun, w0, t, rtol=args.rtol, atol=args.atol, method=args.integrator)
        # wsol shape [len(t), state_dim]
        x1 = wsol[:, 0]
        y1 = wsol[:, 1]
        x2 = wsol[:, 2]
        y2 = wsol[:, 3]
        return x1, y1, x2, y2


def eom(u, p):
    """
    Equations of motion for coupled spring-mass:
    u = [x1, y1, x2, y2]
    p = [m1, m2, k1, k2, L1, L2, b1, b2]
    Returns derivatives [dx1, dy1, dx2, dy2].
    """
    x1 = u[..., 0]
    y1 = u[..., 1]
    x2 = u[..., 2]
    y2 = u[..., 3]
    m1, m2, k1, k2, L1, L2, b1, b2 = p
    dx1 = y1
    dy1 = (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1
    dx2 = y2
    dy2 = (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2
    return dx1, dy1, dx2, dy2


def ic(u, x10, x20):
    ux10 = u[0, 0]
    ux20 = u[0, 2]
    IC_x1 = (ux10 - x10) ** 2
    IC_x2 = (ux20 - x20) ** 2
    return IC_x1, IC_x2


def create_sequential_model(input_size: int = 1, hidden_layers: list = [64, 64],
                            output_size: int = 6, activation=torch.nn.ReLU,
                            dropout_rate: float = 0.2):
    layers = []
    prev_size = input_size
    layers.append(torch.nn.Linear(prev_size, hidden_layers[0]))
    layers.append(activation())
    prev_size = hidden_layers[0]
    for size in (hidden_layers[1:]):
        layers.append(torch.nn.Linear(prev_size, size))
        layers.append(activation())
        prev_size = size
    layers.append(torch.nn.Dropout(dropout_rate))
    layers.append(torch.nn.Linear(prev_size, output_size))
    return torch.nn.Sequential(*layers)


def create_data(args):
    t0 = args.t0
    tN = args.tN
    K = args.K
    t = torch.linspace(t0, tN, 100, device=args.device).to(torch.float32)
    t_test = torch.linspace(t0, tN, 300, device=args.device).to(torch.float32)
    t_grid = torch.linspace(t0, tN, steps=K+1, device=args.device)
    dt = t_grid[1] - t_grid[0]
    t_seg = torch.tensor([0.0, dt], device=args.device)
    return t, t_test, t_grid, dt, t_seg


class MODEL(torch.nn.Module):
    """Neural network for 6-dimensional outputs with enhanced dimension handling"""
    def __init__(self, input_size=1, hidden_size=[64, 64], output_size=6,
                 activation=torch.nn.ReLU, dtype=torch.float32,
                 dropout_rate=0.2):
        super(MODEL, self).__init__()
        self.net = create_sequential_model(input_size=input_size,
                                           hidden_layers=hidden_size,
                                           output_size=output_size,
                                           activation=activation,
                                           dropout_rate=dropout_rate)
        self.n_outputs = output_size
        self.dtype = dtype
    def forward(self, t, u):
        tb = t * torch.ones(u.shape[0], 1, device=u.device)
        inp = torch.cat([u, tb], dim=-1)
        return self.net(inp)
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()


class ODEFunc(torch.nn.Module):
    """Combines the known eom physics with the neural network correction."""
    def __init__(self, model, p):
        super().__init__()
        self.model = model
        self.p = p
    def forward(self, t, u):
        dx1, dy1, dx2, dy2 = eom(u, self.p)
        phys = torch.stack([dx1, dy1, dx2, dy2], dim=-1)
        corr = self.model(t, u)
        return phys + corr


# Trainer class for encapsulating training loop and state
class Trainer:
    def __init__(self, args, data):
        self.args = args
        self.t, self.t_test, self.t_grid, self.dt, self.t_seg = data
        self.t0 = args.t0
        self.tN = args.tN
        self.K = args.K
        self.batch_size = args.batch_size
        self.state_dim = args.state_dim
        self.p = args.p
        self.init_cond = args.init_cond
        # Segment initial states (trainable)
        self.s = torch.nn.Parameter(torch.zeros(self.K, self.batch_size, self.state_dim)).to(args.device)
        # Warm-start segment initial states using the physics-only solution
        with torch.no_grad():
            # Warm-start segment initial states using the physics-only solution
            x1s, y1s, x2s, y2s = solution(self.t_grid, args)
            # stack into [K+1, batch_size, state_dim]
            true_states = torch.stack([x1s, y1s, x2s, y2s], dim=-1).view(self.K+1, self.batch_size, self.state_dim)
            # initialize s[k] for segments 0..K-1
            self.s.data = true_states[:-1].clone()
        # Activation function
        if args.activation_fn == "SiLU":
            activation = torch.nn.SiLU
        else:
            activation = torch.nn.Tanh
        self.model = MODEL(
            input_size=self.state_dim + 1,
            hidden_size=args.hidden_size,
            output_size=self.state_dim,
            activation=activation,
            dropout_rate=args.dropout_rate
        ).to(args.device)
        self.odefunc = ODEFunc(self.model, self.p)
        self.int_options = {'rtol': args.rtol, 'atol': args.atol}
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + [self.s], lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        print("Trainer initialized with:")
        print(f"  Model: {self.model}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Scheduler: {self.scheduler}")
        self.fit_losses = []
        self.cont_losses = []
        self.ic_losses = []
        self.colloc_losses = []
        self.end_losses = []
        self.x1_end_preds = []
        self.x2_end_preds = []

    def train(self):
        args = self.args
        for epoch in range(args.num_epochs):
            self.optimizer.zero_grad()
            u0_batch = self.s.view(self.K * self.batch_size, self.state_dim)
            # ODE solve for all segments
            if args.integrator == 'rk4':
                us = odeint(
                    self.odefunc,
                    u0_batch,
                    self.t_seg,
                    method='rk4'
                )
            else:
                us = odeint(
                    self.odefunc,
                    u0_batch,
                    self.t_seg,
                    rtol=args.rtol,
                    atol=args.atol,
                    method=args.integrator
                )
            u_end = us[1].view(self.K, self.batch_size, self.state_dim)
            # Track predicted endpoint at tN per epoch
            x1_pred_end = u_end[-1, 0, 0].item()
            x2_pred_end = u_end[-1, 0, 2].item()
            self.x1_end_preds.append(x1_pred_end)
            self.x2_end_preds.append(x2_pred_end)
            # Compute overall endpoint loss at tN using the final segment prediction
            t_end_tensor = torch.tensor([self.tN], dtype=torch.float32, device=args.device)
            x1_ts, y1_ts, x2_ts, y2_ts = solution(t_end_tensor, args)
            # Stack full 4D true endpoint
            y_true_end_tensor = torch.stack([x1_ts, y1_ts, x2_ts, y2_ts], dim=0).to(args.device)
            pred_end = u_end[-1, 0]  # shape [state_dim]
            end_loss = (pred_end - y_true_end_tensor).pow(2).mean()
            # Data-fit: true end-points at each segment boundary
            x1_seg, y1_seg, x2_seg, y2_seg = solution(self.t_grid[1:], args)
            y_true_seg = torch.stack([x1_seg, y1_seg, x2_seg, y2_seg], dim=-1).view(self.K, self.batch_size, self.state_dim)
            fit_loss = (u_end - y_true_seg).pow(2).mean()
            cont_loss = (u_end[:-1] - self.s[1:]).pow(2).mean()
            # Initial condition loss
            x10_true, x20_true = args.init_cond[0], args.init_cond[2]
            ic_loss_x1, ic_loss_x2 = ic(self.s[0], x10_true, x20_true)
            ic_loss = ic_loss_x1 + ic_loss_x2
            # Collocation loss at segment midpoints
            t_mid_rel = self.dt / 2.0
            t_mid_tensor = torch.tensor([0.0, t_mid_rel], device=args.device)
            if args.integrator == 'rk4':
                us_mid = odeint(self.odefunc, u0_batch, t_mid_tensor, method='rk4')
            else:
                us_mid = odeint(
                    self.odefunc,
                    u0_batch,
                    t_mid_tensor,
                    rtol=args.rtol,
                    atol=args.atol,
                    method=args.integrator
                )
            u_mid = us_mid[1].view(self.K, self.batch_size, self.state_dim)
            dudt_pred_mid = self.odefunc(t_mid_rel, u_mid.view(self.K * self.batch_size, self.state_dim))
            dudt_pred_mid = dudt_pred_mid.view(self.K, self.batch_size, self.state_dim)
            phys_mid = torch.stack(eom(u_mid, self.p), dim=-1)
            colloc_loss = (dudt_pred_mid - phys_mid).pow(2).mean()
            # Composite loss
            loss = (
                fit_loss
                + args.lambda_cont * cont_loss
                + args.lambda_ic * ic_loss
                + args.lambda_colloc * colloc_loss
                + args.lambda_end * end_loss
            )
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.fit_losses.append(fit_loss.item())
            self.cont_losses.append(cont_loss.item())
            self.ic_losses.append(ic_loss.item())
            self.colloc_losses.append(colloc_loss.item())
            self.end_losses.append(end_loss.item())
            if epoch % args.print_every == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.4e}")
        return (
            self.fit_losses,
            self.cont_losses,
            self.ic_losses,
            self.colloc_losses,
            self.end_losses,
            self.x1_end_preds,
            self.x2_end_preds
        )

# ---- Plotting functions ----
def plot_losses(fit_losses, cont_losses, ic_losses, colloc_losses, end_losses):
    epochs = list(range(len(fit_losses)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.stackplot(
        epochs,
        fit_losses,
        cont_losses,
        ic_losses,
        colloc_losses,
        end_losses,
        labels=['Fit Loss', 'Continuity Loss', 'IC Loss', 'Collocation Loss', 'Endpoint Loss']
    )
    total_losses = [
        f + c + ic + colloc + e
        for f, c, ic, colloc, e in zip(fit_losses, cont_losses, ic_losses, colloc_losses, end_losses)
    ]
    ax1.plot(epochs, total_losses, label='Total Loss', color='black', linewidth=1.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Losses (Stacked)')
    ax1.legend(loc='upper right')
    ax2.plot(epochs, fit_losses,    label='Fit Loss')
    ax2.plot(epochs, cont_losses,   label='Continuity Loss')
    ax2.plot(epochs, ic_losses,     label='IC Loss')
    ax2.plot(epochs, colloc_losses, label='Collocation Loss')
    ax2.plot(epochs, end_losses,    label='Endpoint Loss')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Losses (Individual)')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_endpoint_convergence(x1_end_preds, x2_end_preds, tN, args):
    # Compute true endpoint via physics-only solution
    t_end_tensor = torch.tensor([tN], dtype=torch.float32, device=args.device)
    x1_ts, y1_ts, x2_ts, y2_ts = solution(t_end_tensor, args)
    x1_true_final = x1_ts.item()
    x2_true_final = x2_ts.item()
    epochs_end = list(range(len(x1_end_preds)))
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    ax3.plot(epochs_end, x1_end_preds, label='Predicted x(tN)', color='blue')
    ax3.hlines(x1_true_final, 0, len(epochs_end)-1,
               linestyles='--', label='True x(tN)', color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('x endpoint')
    ax3.set_title('Convergence of x endpoint')
    ax3.legend()
    ax4.plot(epochs_end, x2_end_preds, label='Predicted y(tN)', color='blue')
    ax4.hlines(x2_true_final, 0, len(epochs_end)-1,
               linestyles='--', label='True y(tN)', color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('y endpoint')
    ax4.set_title('Convergence of y endpoint')
    ax4.legend()
    plt.tight_layout()
    plt.show()

def plot_trajectories(odefunc, s0, t_test, t_grid, args):
    with torch.no_grad():
        # NN prediction
        u_pred = odeint(odefunc, s0, t_test, rtol=args.rtol, atol=args.atol, method=args.test_integrator)
        # Remove batch dimension if present: [len, batch, state_dim] -> [len, state_dim]
        if u_pred.ndim == 3:
            u_pred = u_pred.squeeze(1)
        x1_pred = u_pred[:, 0].cpu()
        x2_pred = u_pred[:, 2].cpu()
        # True solution
        x1_true, y1_true, x2_true, y2_true = solution(t_test, args)
        x1_true = x1_true.cpu()
        x2_true = x2_true.cpu()
    t_vals = t_test.cpu()
    # Compute segment boundaries for plotting
    t_seg_cpu = t_grid.cpu()
    x1_seg, _, x2_seg, _ = solution(t_grid, args)
    x1_seg = x1_seg.cpu()
    x2_seg = x2_seg.cpu()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t_vals, x1_true, label='True x1', linestyle='-')
    ax1.plot(t_vals, x1_pred, label='Pred x1', linestyle='--')
    ax1.plot(t_seg_cpu, x1_seg, linestyle='None', marker='o', label='Segment boundaries')
    ax1.set_xlabel('Time'); ax1.set_ylabel('x1'); ax1.set_title('x1: True vs Pred'); ax1.legend()
    ax2.plot(t_vals, x2_true, label='True x2', linestyle='-')
    ax2.plot(t_vals, x2_pred, label='Pred x2', linestyle='--')
    ax2.plot(t_seg_cpu, x2_seg, linestyle='None', marker='o', label='Segment boundaries')
    ax2.set_xlabel('Time'); ax2.set_ylabel('x2'); ax2.set_title('x2: True vs Pred'); ax2.legend()
    plt.tight_layout()
    plt.show()


# ---- Main function ----
def main():
    args = parse_args()
    # Convert activation string to torch.nn module
    data = create_data(args)
    trainer = Trainer(args, data)
    fit_losses, cont_losses, ic_losses, colloc_losses, end_losses, x1_end_preds, x2_end_preds = trainer.train()
    plot_losses(fit_losses, cont_losses, ic_losses, colloc_losses, end_losses)
    plot_endpoint_convergence(x1_end_preds, x2_end_preds, tN=args.tN, args=args)
    plot_trajectories(trainer.odefunc, trainer.s[0], trainer.t_test, trainer.t_grid, args)


if __name__ == "__main__":
    main()