# %% [markdown]
"""Neural ODE with Multi-Shooting for Coupled Spring–Mass System

This script demonstrates a **physics-informed Neural ODE** that integrates known equations of motion for a coupled spring–mass system alongside a small neural network correction. Key features and workflow:

1. **Multi-Shooting Segmentation**  
   - Split the interval \([t_0, t_N]\) into \(K\) segments.  
   - Each segment has its own trainable initial state, warm-started with the analytic solution.  

2. **Physics + Neural Correction**  
   - Define `eom(y)` for the known system dynamics.  
   - Wrap a `MODEL` network to learn residual forces.  
   - Combine them in `ODEFunc.forward(t, y)` and integrate via `torchdiffeq.odeint`.

3. **Composite Loss Function**  
   - **Physics**: enforce the neural ODE to learn the system ODE at the segment mid-points
   - **Data-Fit**: match predicted segment-end states to observed data at boundaries.  
   - **Continuity**: enforce smooth transitions between segments.  
   - **Initial-Condition Penalty**: anchor the first segment to the true start state.  
   - **Collocation**: penalize ODE residual at each segment midpoint.  

4. **Training Setup**  
   - Optimizer: AdamW with weight decay.  
   - LR Scheduler: Cosine annealing over `num_epochs`.  
   - Solver: fixed-step RK4 for efficiency (or adaptive Dormand–Prince).  

5. **Visualization**  
   - **Stacked loss plot**: breakdown of all loss components plus total.  
   - **Endpoint convergence**: predicted vs. true final-state over epochs.  
   - **Trajectory comparison**: true vs. predicted \(x(t)\) & \(y(t)\), with segment-boundary markers.

**Usage:**  
- Adjust `K`, epoch count, and loss weights (`λ_cont`, `λ_ic`, `λ_colloc`) to suit your data.  
- Run the script as-is to train and visualize results.  
- Replace the analytic `solution(t)` with real observations at segment times to fit measured data.
"""
# %% [markdown]
# ## Set up libaries

# %%
import torch
import warnings
from os import cpu_count
from torchdiffeq import odeint
import matplotlib.pyplot as plt
# Add tqdm for progress bar
from tqdm import tqdm
import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Number of *logical* CPUs (hyperthreads included)
n_logical = cpu_count()
torch.set_num_threads(n_logical)
torch.set_num_interop_threads(n_logical)
# import pprint
# pp = pprint.PrettyPrinter(indent=4)

# ---- SIREN/Sine activation ----
class Sine(torch.nn.Module):
    """Sinusoidal activation for SIREN networks."""
    def forward(self, x):
        return torch.sin(x)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train physics-informed Neural ODE on coupled spring-mass system")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_size", type=int, nargs="+", default=[256,128,256,128,256,128])
    parser.add_argument("--activation_fn", type=str, choices=["SiLU", "Tanh", "SIREN"], default="SIREN",
                        help="Activation function: SiLU, Tanh, or SIREN")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--integrator", type=str, choices=["rk4", "dopri5"], default="dopri5")
    parser.add_argument("--test_integrator", type=str, choices=["rk4", "dopri5"], default="rk4")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for dopri5")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for dopri5")
    parser.add_argument("--lambda_cont", type=float, default=1e-1, help="segment continuity loss weight")
    parser.add_argument("--lambda_ic", type=float, default=1e-2, help="initial condition loss weight")
    parser.add_argument("--lambda_colloc", type=float, default=1.0, help="midpoint ODE collocation loss weight")
    parser.add_argument("--lambda_end", type=float, default=1e-2, help="Endpoint loss weight")
    parser.add_argument("--lambda_spec_mag",   type=float, default=0.001, help="Spectral magnitude loss weight")
    parser.add_argument("--lambda_spec_phase", type=float, default=0.001, help="Spectral phase loss weight")
    parser.add_argument("--lambda_spec_mag2",   type=float, default=0.01, help="Spectral magnitude loss weight for x2")
    parser.add_argument("--lambda_spec_phase2", type=float, default=0.001, help="Spectral phase loss weight for x2")
    parser.add_argument("--lambda_energy", type=float, default=1e-3, help="Energy dissipation loss weight")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run")
    parser.add_argument("--print_every", type=int, default=10, help="Print epoch loss values at these intervales")
    parser.add_argument("--t0", type=float, default=0.0, help="Start time")
    parser.add_argument("--tN", type=float, default=10.0, help="End time")
    parser.add_argument("--delta_t", type=float, default=1.0,
                        help="Time decrement for each iteration when truncating the time domain")
    parser.add_argument("--mc_runs", type=int, default=20,
                        help="Number of MC-dropout runs to estimate endpoint uncertainty")
    parser.add_argument(
        "--no-showplot",
        action="store_false",
        dest="showplot",
        default=True,
        help="Disable interactive display of plots"
    )
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


# ---- Energy computation helper ----
def compute_energy(u, p):
    """
    Compute total mechanical energy for coupled spring-mass at states u.
    u shape [...,4] = [x1,y1,x2,y2], p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = u[...,0], u[...,1], u[...,2], u[...,3]
    m1, m2, k1, k2, L1, L2, b1, b2 = p
    E1 = 0.5*m1*y1**2 + 0.5*k1*(x1 - L1)**2
    E2 = 0.5*m2*y2**2 + 0.5*k2*(x2 - x1 - L2)**2
    return E1 + E2


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
    """
    Prepare time grids and compute a single physics-only ODE solve
    over the union of t_grid and t_test, then extract true states
    at both sets of points.
    """
    t0 = args.t0
    tN = args.tN
    K = args.K
    # Segment and test time grids
    t_grid = torch.linspace(t0, tN, steps=K+1, device=args.device).to(torch.float32)
    t_test = torch.linspace(t0, tN, 300, device=args.device).to(torch.float32)
    dt = t_grid[1] - t_grid[0]
    t_seg = torch.tensor([0.0, dt], device=args.device)
    # Build unified time vector without duplicates (strictly increasing)
    all_times = torch.unique(torch.cat([t_grid, t_test]), sorted=True)
    # Physics-only ODE function
    def phys_fun(ti, u):
        dx1, dy1, dx2, dy2 = eom(u, args.p)
        return torch.stack([dx1, dy1, dx2, dy2], dim=-1)
    # Initial condition
    w0 = torch.tensor(args.init_cond, dtype=torch.float32, device=args.device)
    # Single solve
    wsol_all = odeint(phys_fun, w0, all_times,
                       rtol=args.rtol, atol=args.atol,
                       method=args.integrator)
    # wsol_all: [len(all_times), state_dim]
    # Find indices of grid and test times
    grid_idxs = [(all_times == tg).nonzero()[0].item() for tg in t_grid]
    test_idxs = [(all_times == tt).nonzero()[0].item() for tt in t_test]
    # Extract true states
    true_grid = wsol_all[grid_idxs]  # [K+1, state_dim]
    true_test = wsol_all[test_idxs]  # [len(t_test), state_dim]
    # Reshape to include batch dimension
    true_grid_states = true_grid.view(K+1, 1, args.state_dim)
    true_test_states = true_test.view(len(t_test), 1, args.state_dim)

    # Pre-training spectrum analysis on x1 component
    # Flatten true_test_states to 1-D signal for x1
    signal = true_test_states[:, 0, 0].view(-1)
    N = signal.shape[0]
    dt_test = t_test[1] - t_test[0]
    # Compute frequency bins and FFT magnitudes
    freqs = torch.fft.rfftfreq(N, d=dt_test.item())
    fft_vals = torch.abs(torch.fft.rfft(signal))
    # Select dominant frequencies above threshold (10% of max)
    threshold = 0.1 * fft_vals.max()
    dominant_freqs = freqs[fft_vals > threshold]
    fourier_freqs = dominant_freqs

    return t_test, t_grid, dt, t_seg, true_grid_states, true_test_states, fourier_freqs


class MODEL(torch.nn.Module):
    """Neural network for 6-dimensional outputs with enhanced dimension handling and Fourier features"""
    def __init__(self, input_size: int, hidden_size: list, output_size: int,
                 activation: type, dropout_rate: float,
                 fourier_freqs: torch.Tensor):
        super().__init__()
        self.fourier_freqs = fourier_freqs
        # learnable amplitude and phase per Fourier mode
        num_modes = self.fourier_freqs.numel()
        self.phi = torch.nn.Parameter(torch.zeros(num_modes, device=fourier_freqs.device))
        self.A   = torch.nn.Parameter(torch.ones(num_modes, device=fourier_freqs.device))
        self.net = create_sequential_model(input_size=input_size,
                                           hidden_layers=hidden_size,
                                           output_size=output_size,
                                           activation=activation,
                                           dropout_rate=dropout_rate)
        # Debug: print first layer weight shape
        # first_layer = self.net[0]
        # if isinstance(first_layer, torch.nn.Linear):
        #     print(f"[DEBUG] First Linear layer weight shape = {first_layer.weight.shape}")
        self.n_outputs = output_size
    def forward(self, t, u):
        batch = u.shape[0]
        # Time embedding: ensure tb shape is [batch, 1]
        if t.dim() == 0:
            tb = t * torch.ones(batch, 1, device=u.device)
        elif t.dim() == 1:
            tb = t.unsqueeze(1)
        else:
            tb = t
        # Fourier features with learnable amplitude A and phase phi
        feats = []
        for i, f in enumerate(self.fourier_freqs):
            # compute phase-shifted, amplitude-scaled sine feature
            fb = tb * (2 * torch.pi * f) + self.phi[i]
            feats.append(self.A[i].unsqueeze(0) * torch.sin(fb))
        # concatenate features of shape [batch, num_modes]
        fourier_feats = torch.cat(feats, dim=1) if feats else torch.zeros((batch, 0), device=u.device)
        inp = torch.cat([u, tb, fourier_feats], dim=1)
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


class Trainer:
    def __init__(self, args, data):
        self.args = args
        self.t_test, self.t_grid, self.dt, self.t_seg, true_grid_states, true_test_states, fourier_freqs = data
        self.t0 = args.t0
        self.tN = args.tN
        self.K = args.K
        self.batch_size = args.batch_size
        self.state_dim = args.state_dim
        self.p = args.p
        self.init_cond = args.init_cond
        self.true_grid_states = true_grid_states
        self.true_test_states = true_test_states
        self.fourier_freqs = fourier_freqs.to(args.device)
        # Segment initial states (trainable)
        self.s = torch.nn.Parameter(torch.zeros(self.K, self.batch_size, self.state_dim)).to(args.device)
        # Initialize segment states from true physics-only solution at t_grid
        with torch.no_grad():
            self.s.data = true_grid_states[:-1].clone() # [K+1, 1, state_dim]
        # Activation function
        if args.activation_fn == "SiLU":
            activation = torch.nn.SiLU
        else:
            if args.activation_fn == "Tanh":
                activation = torch.nn.Tanh
            else:
                activation = Sine
        input_size = self.state_dim + 1 + self.fourier_freqs.shape[0]
        # Debug: print computed input size and Fourier frequencies count
        # print(f"[DEBUG] input_size = {input_size}, fourier_freqs count = {self.fourier_freqs.shape[0]}")
        self.model = MODEL(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=self.state_dim,
            activation=activation,
            dropout_rate=args.dropout_rate,
            fourier_freqs=self.fourier_freqs
        ).to(args.device)
        self.odefunc = ODEFunc(self.model, self.p)
        self.int_options = {'rtol': args.rtol, 'atol': args.atol}
        # Separate phi/A parameters for higher LR and no weight decay
        phiA_params = [self.model.phi, self.model.A]
        # All other model params (excluding phi/A) + segment states
        other_params = [p for p in self.model.parameters() if not any(p is q for q in phiA_params)]
        other_params.append(self.s)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": phiA_params, "lr": 1e-2, "weight_decay": 0},
                {"params": other_params, "lr": 1e-3, "weight_decay": 1e-4}
            ]
        )
        # Two-stage LR: drop LR by 10x at halfway and 3/4 of training
        # LR drop at the end of the spectral ramp, then again later
        # ramp_epochs = args.num_epochs // 2
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=[ramp_epochs, 3 * args.num_epochs // 4],
        #     gamma=0.1
        #     )
        # instead of MultiStepLR, do
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=1e-5
        )
        # print("Trainer initialized with:")
        # print(f"  Model: {self.model}")
        # print(f"  Optimizer: {self.optimizer}")
        # print(f"  Scheduler: {self.scheduler}")
        self.fit_losses = []
        self.cont_losses = []
        self.ic_losses = []
        self.colloc_losses = []
        self.end_losses = []
        self.total_losses = []
        self.x1_end_preds = []
        self.x2_end_preds = []
        self.spec_mag_losses = []
        self.spec_phase_losses = []
        self.spec_mag_losses2 = []
        self.spec_phase_losses2 = []
        self.n_colloc = 5  # number of interior collocation points
        self.incremental_results = None

    def train_incremental(self, delta_t, mc_runs=20):
        """ method that:
        •	Iteratively truncates the end time,
        •	Retrains from scratch,
        •	Captures endpoint errors and their uncertainty via MC-dropout,
        •	Stores all results in self.incremental_results.

        updated train_incremental so that for each truncated training:
        •	MC-dropout and continuous prediction both integrate over the 
            original t_test (full domain), using self.t_test.
        •	Errors (err1, err2) are now computed against the true endpoints at 
            the original tN (self.true_test_states[-1]).
        •	We keep the same structure for recording uncertainty and results.

        This way, even though the model trains on reduced data, the evaluation 
        always measures performance at the full end time.
        
        Two new supporting plotting functions:
        •	plot_error_vs_time for error vs. time-removed with uncertainty bands.
        •	plot_error_surface that shows a 3D surface and heatmap of per-epoch 
            endpoint loss across truncations.
        """
        t0 = self.args.t0
        original_tN = self.args.tN
        current_tN = original_tN
        results = []
        # Set up progress bar for incremental truncation steps
        total_steps = int(np.floor((original_tN - t0) / delta_t)) + 1
        pbar = tqdm(total=total_steps, desc='Incremental Training', unit='step')
        while current_tN >= t0 + delta_t:
            # Prepare truncated arguments and data
            args_loc = copy.deepcopy(self.args)
            args_loc.tN = current_tN
            data_loc = create_data(args_loc)
            # Train fresh model
            tr = Trainer(args_loc, data_loc)
            tr.train()
            # Ensure dropout is active for MC sampling
            if tr.args.dropout_rate == 0:
                warnings.warn("dropout_rate=0; MC-dropout uncertainty will be zero. Set --dropout_rate > 0 to enable variability.")
            tr.model.train()
            tr.model.enable_dropout()
            mc_preds1 = []
            mc_preds2 = []
            # Use the full t_test for MC-dropout (original tN)
            full_t_test = self.t_test.to(self.args.device)
            for _ in range(mc_runs):
                u_pred = odeint(tr.odefunc, tr.s[0], full_t_test,
                                rtol=tr.args.rtol, atol=tr.args.atol,
                                method=tr.args.test_integrator)
                if u_pred.ndim == 3:
                    u_pred = u_pred.squeeze(1)
                mc_preds1.append(u_pred[-1, 0].item())
                mc_preds2.append(u_pred[-1, 2].item())
            # Compute uncertainty of the endpoint error via MC-dropout samples
            # Error samples = predicted sample minus true endpoint
            x1_true_full = self.true_test_states[-1, 0, 0].item()
            x2_true_full = self.true_test_states[-1, 0, 2].item()
            err_samples1 = [pred - x1_true_full for pred in mc_preds1]
            err_samples2 = [pred - x2_true_full for pred in mc_preds2]
            unc1 = float(np.std(err_samples1))
            unc2 = float(np.std(err_samples2))
            # Continuous prediction at the original end time (tN)
            u_full = odeint(tr.odefunc, tr.s[0], full_t_test,
                            rtol=tr.args.rtol, atol=tr.args.atol,
                            method=tr.args.test_integrator)
            if u_full.ndim == 3:
                u_full = u_full.squeeze(1)
            x1_pred_full = u_full[-1, 0].item()
            x2_pred_full = u_full[-1, 2].item()

            # True endpoints at original tN from original data
            x1_true_full = self.true_test_states[-1, 0, 0].item()
            x2_true_full = self.true_test_states[-1, 0, 2].item()

            # Errors relative to the original end time
            err1 = abs(x1_pred_full - x1_true_full)
            err2 = abs(x2_pred_full - x2_true_full)
            # Record results
            results.append({
                'tN_truncated': current_tN,
                'x1_error': err1,
                'x2_error': err2,
                'x1_unc': unc1,
                'x2_unc': unc2,
                'end_losses': tr.end_losses
            })
            current_tN -= delta_t
            pbar.update(1)
        pbar.close()
        self.incremental_results = results
        return results

    def train(self):
        args = self.args
        # Early‐stop and freeze setup for phase parameters
        ramp_epochs = args.num_epochs // 2
        true_grid_states = self.true_grid_states
        true_test_states = self.true_test_states
        pbar = tqdm(range(args.num_epochs), desc='Training', unit='epoch')
        for epoch in pbar:
            # Reset gradients
            self.optimizer.zero_grad()
            # Flatten segment initial states into batch for ODE solver
            u0_batch = self.s.view(self.K * self.batch_size, self.state_dim)
            # Solve neural ODE across each segment using chosen integrator
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
            # Reshape solver output to [K segments, batch, state_dim]
            u_end = us[1].view(self.K, self.batch_size, self.state_dim)
            # Record predicted endpoint values x1 and x2 for the final segment
            x1_pred_end = u_end[-1, 0, 0].item()
            x2_pred_end = u_end[-1, 0, 2].item()
            self.x1_end_preds.append(x1_pred_end)
            self.x2_end_preds.append(x2_pred_end)
            # Compute true physics-only endpoint at final time for loss
            y_true_end_tensor = true_test_states[-1, 0]  # [state_dim]
            pred_end = u_end[-1, 0]  # shape [state_dim]
            end_loss = (pred_end - y_true_end_tensor).pow(2).mean()

            # Anneal spectral-loss weight: start at 0, ramp to full over first half
            if epoch < ramp_epochs:
                spec_weight = epoch / ramp_epochs
            else:
                spec_weight = 1.0
            λ_spec_mag = spec_weight * args.lambda_spec_mag
            λ_spec_phase = spec_weight * args.lambda_spec_phase
            λ_spec_mag2 = spec_weight * args.lambda_spec_mag2
            λ_spec_phase2 = spec_weight * args.lambda_spec_phase2
            # ----- Spectral loss on x1 trajectory -----
            # True and predicted x1 signals
            with torch.no_grad():
                u_pred = odeint(self.odefunc, self.s[0], self.t_test, rtol=args.rtol, atol=args.atol, method=args.test_integrator)
                if u_pred.ndim == 3:
                    u_pred = u_pred.squeeze(1)
            true_signal = self.true_test_states[:, 0, 0].view(-1)
            pred_signal = u_pred[:, 0]
            # FFTs
            fft_true = torch.fft.rfft(true_signal)
            fft_pred = torch.fft.rfft(pred_signal)
            # magnitude & phase
            mag_true   = torch.abs(fft_true)
            mag_pred   = torch.abs(fft_pred)
            phase_true = torch.atan2(fft_true.imag, fft_true.real)
            phase_pred = torch.atan2(fft_pred.imag, fft_pred.real)
            # Add epsilon for log-magnitude stability
            eps = 1e-6
            # Log-magnitude spectral loss to reduce dynamic range blow-up
            spec_mag_loss   = (torch.log(mag_pred + eps) - torch.log(mag_true + eps)).pow(2).mean()
            spec_phase_loss = (phase_pred - phase_true).pow(2).mean()
            # track for plotting/analysis
            self.spec_mag_losses.append(spec_mag_loss.item())
            self.spec_phase_losses.append(spec_phase_loss.item())
            # optionally, print top-3 frequency mismatches at first epoch
            # if epoch == 0:
                # N = true_signal.shape[0]
                # dt_test = self.t_test[1] - self.t_test[0]
                # freqs = torch.fft.rfftfreq(N, d=dt_test.item())
                # magnitude error
                # mag_err = (mag_pred - mag_true).abs()
                # topk = torch.topk(mag_err, k=min(3, mag_err.numel()))
                # top_freqs = freqs[topk.indices]
                # print(f"[DEBUG] Top magnitude error freqs: {top_freqs.tolist()}")
            # ----- Spectral loss on x2 trajectory -----
            true_signal2 = self.true_test_states[:, 0, 2].view(-1)
            pred_signal2 = u_pred[:, 2]
            fft_true2 = torch.fft.rfft(true_signal2)
            fft_pred2 = torch.fft.rfft(pred_signal2)
            mag_true2   = torch.abs(fft_true2)
            mag_pred2   = torch.abs(fft_pred2)
            phase_true2 = torch.atan2(fft_true2.imag, fft_true2.real)
            phase_pred2 = torch.atan2(fft_pred2.imag, fft_pred2.real)
            # Log-magnitude spectral loss for x2
            spec_mag_loss2   = (torch.log(mag_pred2 + eps) - torch.log(mag_true2 + eps)).pow(2).mean()
            spec_phase_loss2 = (phase_pred2 - phase_true2).pow(2).mean()
            self.spec_mag_losses2.append(spec_mag_loss2.item())
            self.spec_phase_losses2.append(spec_phase_loss2.item())
            # 1) Freeze φ/A once the spectral‐loss ramp is done
            if epoch == ramp_epochs:
                print(f"[INFO] Freezing spectral-shift (phi) and amplitude (A) at epoch {epoch}")
                self.model.phi.requires_grad = False
                self.model.A.requires_grad = False
            # Compute data-fit loss matching segment boundaries
            y_true_seg = true_grid_states[1:]  # [K,1,state_dim]
            fit_loss = (u_end - y_true_seg).pow(2).mean()
            # Compute continuity loss between adjacent segment end and next segment start
            cont_loss = (u_end[:-1] - self.s[1:]).pow(2).mean()
            # Compute initial condition penalty for first segment
            x10_true, x20_true = args.init_cond[0], args.init_cond[2]
            ic_loss_x1, ic_loss_x2 = ic(self.s[0], x10_true, x20_true)
            ic_loss = ic_loss_x1 + ic_loss_x2
            # Compute collocation loss at multiple points enforcing ODE residual
            # sample points in (0, dt) excluding endpoints
            t_colloc = torch.linspace(0.0, self.dt, steps=self.n_colloc+2, device=args.device)[1:-1]
            # solve neural ODE at all collocation times
            if args.integrator == 'rk4':
                us_colloc = odeint(self.odefunc, u0_batch, t_colloc, method='rk4')
            else:
                us_colloc = odeint(
                    self.odefunc,
                    u0_batch,
                    t_colloc,
                    rtol=args.rtol,
                    atol=args.atol,
                    method=args.integrator
                )
            # us_colloc: [len(t_colloc), batch_size*K, state_dim]
            # compute neural-ODE predicted derivatives at collocation points
            # flatten u_colloc to [len*K*batch, state_dim]
            u_colloc_flat = us_colloc.view(-1, self.state_dim)
            # repeat t_colloc for each segment and batch to match u_colloc_flat
            t_flat = t_colloc.repeat_interleave(self.K * self.batch_size)
            # get derivatives in one call: shape [len*K*batch, state_dim]
            dudt_flat = self.odefunc(t_flat, u_colloc_flat)
            # reshape back to [len, K, batch, state_dim]
            dudt_pred = dudt_flat.view(len(t_colloc), self.K, self.batch_size, self.state_dim)
            # compute true physics derivatives at collocation points
            u_colloc = us_colloc.view(len(t_colloc), self.K, self.batch_size, self.state_dim)
            phys_colloc = torch.stack(eom(u_colloc, self.p), dim=-1)  # [len, K, batch, state_dim]
            # mean squared residual over all collocation points
            colloc_loss = (dudt_pred - phys_colloc).pow(2).mean()
            # Compute energy-dissipation regularization
            # Solve for predicted trajectory with gradients
            u_pred_grad = odeint(self.odefunc, self.s[0], self.t_test,
                                 rtol=args.rtol, atol=args.atol,
                                 method=args.test_integrator)
            if u_pred_grad.ndim == 3:
                u_pred_grad = u_pred_grad.squeeze(1)
            # Compute energy at each time step
            E = compute_energy(u_pred_grad, self.p)
            E_next = compute_energy(u_pred_grad[1:], self.p)
            energy_loss = torch.relu(E_next - E[:-1]).mean()
            # Combine all loss components into total loss (excluding end_loss, which is tracked separately)
            loss = (
                fit_loss
                + args.lambda_cont * cont_loss
                + args.lambda_ic * ic_loss
                + args.lambda_colloc * colloc_loss
                # end_loss is computed and tracked but not included in the training loss
                + λ_spec_mag   * spec_mag_loss
                + λ_spec_phase * spec_phase_loss
                + λ_spec_mag2   * spec_mag_loss2
                + λ_spec_phase2 * spec_phase_loss2
                + args.lambda_energy * energy_loss
            )
            # Backpropagate total loss
            loss.backward()
            # Clip gradients to prevent exploding updates
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [self.s],
                max_norm=1.0
            )
            # Update model and segment states
            self.optimizer.step()
            # Update learning rate schedule
            self.scheduler.step()
            # Update tqdm with loss
            pbar.set_postfix(loss=loss.item())
            self.total_losses.append(loss.item())
            self.fit_losses.append(fit_loss.item())
            self.cont_losses.append(cont_loss.item())
            self.ic_losses.append(ic_loss.item())
            self.colloc_losses.append(colloc_loss.item())
            self.end_losses.append(end_loss.item())
        # Print summarized losses at specified intervals
        # self.print_loss_table()
        return (
            self.fit_losses,
            self.cont_losses,
            self.ic_losses,
            self.colloc_losses,
            self.end_losses,
            self.x1_end_preds,
            self.x2_end_preds,
            self.spec_mag_losses,
            self.spec_phase_losses,
            self.spec_mag_losses2,
            self.spec_phase_losses2
        )

    def print_loss_table(self):
        loss_dict = {
            'Fit': self.fit_losses,
            'Cont': self.cont_losses,
            'IC': self.ic_losses,
            'Colloc': self.colloc_losses,
            'End': self.end_losses,
            'Total': self.total_losses
        }
        # Determine order of magnitude for each loss type
        exps = {}
        for name, losses in loss_dict.items():
            max_val = max(losses) if losses else 0
            exp = int(np.floor(np.log10(max_val))) if max_val > 0 else 0
            exps[name] = exp
        # Build header with scale factors
        header = f"{'Epoch':>5}"
        for name in loss_dict.keys():
            # Align header labels to match 13-character data columns (space + 12 width)
            header += f"{name}(1e{exps[name]:d})".rjust(13)
        print("\nLoss summary:")
        print(header)
        # Print rows at intervals
        for i in range(len(self.total_losses)):
            if i % self.args.print_every == 0:
                row = f"{i:5d}"
                for name, losses in loss_dict.items():
                    scale = 10 ** exps[name] if exps[name] != 0 else 1
                    val = losses[i] / scale
                    row += f" {val:12.3f}"
                print(row)

# ---- Plotting functions ----
def plot_losses(fit_losses, cont_losses, ic_losses, colloc_losses, end_losses, showplot=True):
    epochs = list(range(len(fit_losses)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.stackplot(
        epochs,
        fit_losses,
        cont_losses,
        ic_losses,
        colloc_losses,
        end_losses,
        labels=['Fit Loss', 'Continuity Loss', 'IC Loss', 'Collocation (EOM) Loss', 'Endpoint Loss']
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
    ax2.plot(epochs, colloc_losses, label='Collocation (EOM) Loss')
    ax2.plot(epochs, end_losses,    label='Endpoint Loss')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Losses (Individual)')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("training_losses.png")
    if showplot:
        plt.show()

def plot_endpoint_convergence(x1_end_preds, x2_end_preds, t_test, args, showplot=True):
    # Compute true solution over the full t_test grid and pick the final point
    t_test_tensor = t_test.to(args.device)
    x1_ts, y1_ts, x2_ts, y2_ts = solution(t_test_tensor, args)
    x1_true_final = x1_ts[-1].item()
    x2_true_final = x2_ts[-1].item()
    epochs_end = list(range(len(x1_end_preds)))
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    ax3.plot(epochs_end, x1_end_preds, label='Predicted x(tN)', color='blue')
    ax3.hlines(x1_true_final, 0, len(epochs_end)-1,
               linestyles='--', label='True x1(tN)', color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('x1 endpoint')
    ax3.set_title('Convergence of x1 endpoint')
    ax3.legend()
    ax4.plot(epochs_end, x2_end_preds, label='Predicted x2(tN)', color='blue')
    ax4.hlines(x2_true_final, 0, len(epochs_end)-1,
               linestyles='--', label='True x2(tN)', color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('x2 endpoint')
    ax4.set_title('Convergence of x2 endpoint')
    ax4.legend()
    plt.tight_layout()
    plt.savefig("endpoint_convergence.png")
    if showplot:
        plt.show()

def plot_trajectories(odefunc, s, t_test, t_grid, args, showplot=True):
    """ compare predicted and true trajectories

    Args:
        odefunc:  Neural ODE estimated solution of the summation of the
                  physics ODE and the neural network ODE model
        s (tensor): segmented data of shape [K, batch_size, state_dim] where
                  the data is represented by K+1 segments
        t_test (tensor): independent time test data, e.g. linspace(t0, tN, 300)
        t_grid (tensor): independent time segment data, e.g. linspace(t0, tN, steps=K+1)

    NOTE:  This is how I get u_pred from odefunc, s, t?
    When you call
        u_pred = odeint(odefunc, s[0], t_test, …)
    the shape of the returned tensor is actually
        [ len(t_test), batch_size, state_dim ]
    —not [time, x1, y1, x2, y2] directly.  
    
    Here's why:
        1.	Input to odeint
            You passed in s[0] as the “initial state” for the network.  s is a 
            tensor of shape [K, batch_size, state_dim], so s[0] has shape 
            [batch_size, state_dim].  In your default setup batch_size == 1, 
            and state_dim == 4 (for [x1, y1, x2, y2]).

        2.	odeint output shape
            By design, torchdiffeq.odeint(func, y0, t) returns a tensor of shape
            [ len(t), *y0.shape ] so here that becomes 
            [ len(t_test), batch_size, state_dim ]

            Concretely:
                •	Dimension 0 indexes time points
                •	Dimension 1 indexes batch examples (you only have one)
                •	Dimension 2 indexes the state variables (x1, y1, x2, y2)

        3.	Why check u_pred.ndim == 3?
            Since you typically work with a single batch (batch_size=1), you 
            often don’t want that extra singleton dimension in your plotting 
            code.  The check

            if u_pred.ndim == 3:
                u_pred = u_pred.squeeze(1)

            collapses the shape from [len, 1, 4] down to [len, 4].  After that, 
            you can index directly:

            x1_pred = u_pred[:,0]  # all times, state index 0 (x1)
            x2_pred = u_pred[:,2]  # all times, state index 2 (x2)

    So in summary:
        •	Yes, it does contain all four state variables per time point, but 
        wrapped inside a batch dimension.
        •	The ndim == 3 guard simply strips out that batch axis when it's 
        length 1 so that your downstream code can treat u_pred as a 2-D array 
        of shape [time, state_dim].
    """
    with torch.no_grad():
        # NN prediction
        u_pred = odeint(odefunc, s[0], t_test, rtol=args.rtol, atol=args.atol, method=args.test_integrator)
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
    # Model-predicted segment nodes from learned initial states
    t_seg_cpu = t_grid.cpu()
    s_cpu = s.detach().cpu()  # [K, batch, state_dim]
    x1_nodes = s_cpu[:, 0, 0]
    x2_nodes = s_cpu[:, 0, 2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t_vals, x1_true, label='True x1', linestyle='-')
    ax1.plot(t_vals, x1_pred, label='Pred x1', linestyle='--')
    ax1.plot(t_seg_cpu[:-1], x1_nodes, linestyle='None', marker='o', label='Model segment nodes')
    ax1.set_xlabel('Time'); ax1.set_ylabel('x1'); ax1.set_title('x1: True vs Pred'); ax1.legend()
    ax2.plot(t_vals, x2_true, label='True x2', linestyle='-')
    ax2.plot(t_vals, x2_pred, label='Pred x2', linestyle='--')
    ax2.plot(t_seg_cpu[:-1], x2_nodes, linestyle='None', marker='o', label='Model segment nodes')
    ax2.set_xlabel('Time'); ax2.set_ylabel('x2'); ax2.set_title('x2: True vs Pred'); ax2.legend()
    plt.tight_layout()
    plt.savefig("trajectories_comparison.png")
    if showplot:
        plt.show()

def plot_spectral_losses(spec_mag_losses, 
                         spec_phase_losses, 
                         spec_mag_losses2, 
                         spec_phase_losses2,
                         showplot=True):
    # Optionally plot spectral losses
    plt.figure()
    plt.plot(spec_mag_losses, label="Spectral Magnitude Loss (x1)")
    plt.plot(spec_phase_losses, label="Spectral Phase Loss (x1)")
    plt.plot(spec_mag_losses2, label="Spectral Magnitude Loss (x2)")
    plt.plot(spec_phase_losses2, label="Spectral Phase Loss (x2)")
    plt.yscale("log")
    plt.legend()
    plt.title("Spectral Losses")
    plt.savefig("spectral_losses.png")
    if showplot:
        plt.show()


# ---- New plotting functions for incremental study ----
def plot_error_vs_time(results, original_tN, showplot=True):
    removed = [original_tN - rec['tN_truncated'] for rec in results]
    errors1 = [rec['x1_error'] for rec in results]
    errors2 = [rec['x2_error'] for rec in results]
    uncs1 = [rec['x1_unc'] for rec in results]
    uncs2 = [rec['x2_unc'] for rec in results]
    data = dict(
        {'Δt': removed,
         'e1': errors1,
         'e2': errors2,
         'δ1': uncs1,
         'δ2': uncs2}
    )
    
    fig, ax = plt.subplots()
    ax.errorbar(removed, errors1, yerr=uncs1, label='x1 Error', fmt='-o')
    ax.errorbar(removed, errors2, yerr=uncs2, label='x2 Error', fmt='-o')
    ax.set_xlabel('Time Removed')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Endpoint Error vs. Time Removed')
    ax.legend()
    plt.savefig("endpoint_error_vs_time_removed.png")
    if showplot:
        plt.show()
    print("Error data:")
    # Print the names of the columns.
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Δt', 'e1', 'δ1', 'e2', 'δ2'))
    # print each data row.
    for dt, e1, e2, u1, u2 in zip(data['Δt'], data['e1'], data['e2'], data['δ1'], data['δ2']):
        print("{:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(dt, e1, u1, e2, u2))

def plot_error_surface(results, original_tN, showplot=True):
    t_removed = np.array([original_tN - rec['tN_truncated'] for rec in results])
    end_losses = np.array([rec['end_losses'] for rec in results])
    epochs = np.arange(end_losses.shape[1])
    # Create grid for surface
    T, E = np.meshgrid(t_removed, epochs, indexing='ij')

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(T, E, end_losses, cmap='viridis')
    ax1.set_xlabel('Time Removed')
    ax1.set_ylabel('Epoch')
    ax1.set_zlabel('Endpoint Loss')
    ax1.set_title('3D Surface of Endpoint Loss')

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(end_losses, aspect='auto', origin='lower',
                    extent=[t_removed.min(), t_removed.max(),
                            epochs.min(), epochs.max()])
    ax2.set_xlabel('Time Removed')
    ax2.set_ylabel('Epoch')
    ax2.set_title('Heatmap of Endpoint Loss')
    fig.colorbar(im, ax=ax2, label='Endpoint Loss')

    plt.tight_layout()
    plt.savefig("endpoint_error_surface_heatmap.png")
    if showplot:
        plt.show()

# ---- Main function ----
def main():
    args = parse_args()
    data = create_data(args)
    trainer = Trainer(args, data)
    print("Trainer initialized with:")
    print(f"  Model: {trainer.model}")
    print(f"  Optimizer: {trainer.optimizer}")
    print(f"  Scheduler: {trainer.scheduler}")
    fit_losses, cont_losses, ic_losses, colloc_losses, end_losses, x1_end_preds, x2_end_preds, spec_mag_losses, spec_phase_losses, spec_mag_losses2, spec_phase_losses2 = trainer.train()
    # Print summarized losses at specified intervals
    trainer.print_loss_table()
    plot_losses(fit_losses, cont_losses, ic_losses, colloc_losses, end_losses, showplot=args.showplot)
    plot_endpoint_convergence(x1_end_preds, x2_end_preds, t_test=trainer.t_test, args=args, showplot=args.showplot)
    plot_trajectories(trainer.odefunc, trainer.s, trainer.t_test, trainer.t_grid, args, showplot=args.showplot)
    plot_spectral_losses(spec_mag_losses, spec_phase_losses, spec_mag_losses2, spec_phase_losses2, showplot=args.showplot)

    # Incremental truncation study
    results = trainer.train_incremental(args.delta_t, args.mc_runs)
    plot_error_vs_time(results, args.tN, showplot=args.showplot)
    plot_error_surface(results, args.tN, showplot=args.showplot)


if __name__ == "__main__":
    main()