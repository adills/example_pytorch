"""GAN for synthetic aircraft time series data.

Author: AN Dills
Date: 2024-06-01

Situation:
Suppose we have a time series dataset of aircraft flight parameters
(e.g., altitude, heading, latitude, longitude) recorded at regular intervals
during flights. We want to use a Generative Adversarial Network (GAN) to
generate synthetic time series data that mimics the real flight data.

Dataset:
OpenSky Network provides a dataset of flight trajectories, which includes
time series. We can use the OpenSky REST API to fetch flight trajectories
and then preprocess the data to create a suitable dataset for training a GAN.
Using the sister module `opensky_example.py`, we can fetch and preprocess the
flight trajectory by reading in the CSV file, which has data that was filtered
for long-haul flights, and then fetching the trajectories.

Source data file CSV: `long_distance_trajectories.csv`

The CSV file contains the following columns:
- time: timestamp of the recorded data point
- lat: latitude
- lon: longitude
- altitude_m: altitude in meters
- true_track_deg: true track in degrees
- onground: boolean indicating if the aircraft is on the ground
- icao24: The unique ICAO 24-bit address of the aircraft
- callsign: The flight's callsign
- origin: The departure airport
- destination: The arrival airport
- first_seen: The timestamp of the first recorded position
- last_seen: The timestamp of the last recorded position
- duration_hours: The duration of the flight in hours

TASKS:
1. Load the dataset and preprocess it to create sequences of time series data.
   1.1 Load and parse the trajectory CSV.
   1.2 Group rows into flight-wise sequences ordered by time.
   1.3 Convert raw values into torch tensors without NumPy.
   1.4 Normalize features and create sliding windows for GAN training.
   1.5 Validate preprocessing outputs before model construction.
2. Define a GAN architecture suitable for time series data.
   2.1 Generator: convert the article's dense generator to PyTorch and adapt
       it to emit full time-series windows.
   2.2 Discriminator: convert the article's dense discriminator to PyTorch and
       adapt it to score full time-series windows.
   2.3 Assemble the GAN training components and validate tensor shapes.
3. Train the GAN on the preprocessed dataset.
   3.1 Train the discriminator on real and generated batches.
   3.2 Train the generator to fool the discriminator.
   3.3 Run the adversarial loop and validate the loss history.
4. Generate synthetic time series data and evaluate its quality.
   4.1 Sample synthetic sequences from the trained generator.
   4.2 Invert normalization back to physical units.
   4.3 Evaluate distribution, trajectory, and temporal quality metrics.
   4.4 Optionally save comparison plots for qualitative inspection.
5. Reconstruct complete synthetic trajectories from generated windows and compare
   them against the original airborne training flights.
   5.1 Keep window metadata so sampled windows retain their own elapsed-time axis.
   5.2 Rebuild full synthetic flights on each retained flight's elapsed-time grid
       by generating overlapping windows and averaging them back together.
   5.3 Plot original altitude trajectories overlaid with synthetic altitude
       trajectories on elapsed-time axes.
   5.4 Plot the mean training altitude trajectory with its 1-sigma envelope
       overlaid with the mean synthetic altitude trajectory and its 1-sigma
       envelope on a shared normalized-time grid.

Current Version (May 26, 2026):
A. `gan_timeseries_data.py` is now filled in as a full PyTorch GAN scaffold at 
[gan_timeseries_data.py](/Users/anthonydills/gitprojects/example_pytorch/gan_timeseries_data.py), 
with the original 4 major TASK sections preserved and the Medium article's 
outline inserted as ordered sub-TASKs under each one.

B. The implementation stays off NumPy in the data and training path. It loads 
the OpenSky CSV with pandas, builds torch tensors for the learned state 
features, creates sliding flight windows, defines a PyTorch 
generator/discriminator pair for sequence synthesis, trains them with an 
adversarial loop, and evaluates synthetic sequences with trajectory RMSE, 
feature mean/std gaps, and lag-1 
autocorrelation gaps. Each major TASK also has a validation checkpoint so 
preprocessing, model shapes, and training history are checked before proceeding.

C. I switched verification to `pipenv run` as requested. 
`pipenv run python -m py_compile gan_timeseries_data.py` passed, and repeated 
1-epoch smoke tests also ran successfully on the local CSV.

D. The current script treats time as a fixed normalized conditioning channel
rather than a generated feature, uses `onground` only as a filtering label so
the GAN learns airborne state trajectories, and reconstructs complete synthetic
flights by overlap-averaging generated windows back onto retained flight time
grids.

NOTE: Regarding Step 1.4, A flight can be long and every flight has a different 
number of points. The GAN needs many training examples with one consistent 
shape, so we cut each flight into fixed-length subsequences. In this script, 
each window has length sequence_length=32 by default. Windows are extracted 
with overlap using stride=8. After the latest changes, each window is shape (32, 5): 
lat, lon, altitude_m, track_sin, track_cos.  
*How the windows are used*:
- create_training_windows(...) turns full flights into many normalized windows 
for training by cutting each flight into overlapping (sequence_length, 
feature_dim) windows.
- Those windows are stacked into one tensor and fed to the DataLoader for the 
actual GAN training examples.
- During training, the discriminator sees real windows from that tensor and 
fake windows from the generator.
- The generator does not generate a whole flight. It generates one fixed-length 
window at a time, conditioned on normalized time.
*Why this is useful*:
- It increases the number of training samples from a small number of flights.
- It makes training feasible because every sample has the same dimensions.
- It lets the model learn local trajectory behavior such as climb/cruise/turn patterns without needing to model an entire multi-hour flight at once.
*Step 5 solution*:
- The script now carries per-window metadata so sampled windows keep their exact
elapsed-time axes for evaluation plots.
- After training, a trajectory assembly step generates overlapping synthetic
windows for each retained flight length and averages those windows back into one
complete synthetic trajectory on the original elapsed-time grid.
- Two additional plots summarize the result:
  original vs synthetic altitude trajectories, and mean altitude with 1-sigma
  envelopes on a shared normalized-time grid.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path(__file__).with_name("long_distance_trajectories.csv")
FEATURE_NAMES = (
    "lat",
    "lon",
    "altitude_m",
    "track_sin",
    "track_cos",
)


@dataclass
class DataConfig:
    sequence_length: int = 32
    stride: int = 8
    min_flight_points: int = 48
    batch_size: int = 32


@dataclass
class ModelConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 1


@dataclass
class TrainConfig:
    epochs: int = 250
    generator_lr: float = 1e-4
    discriminator_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.9
    print_every: int = 25


@dataclass
class NormalizationStats:
    mean: Tensor
    std: Tensor


@dataclass
class WindowMetadata:
    flight_label: str
    flight_index: int
    start_index: int
    end_index: int
    elapsed_seconds: Tensor


@dataclass
class FullTrajectory:
    label: str
    elapsed_seconds: Tensor
    normalized_time: Tensor
    features: Tensor


@dataclass
class GANComponents:
    generator: nn.Module
    discriminator: nn.Module
    generator_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer
    criterion: nn.Module


def build_normalized_time_channel(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    time_steps = torch.linspace(
        0.0,
        1.0,
        steps=sequence_length,
        device=device,
        dtype=dtype,
    )
    return time_steps.view(1, sequence_length, 1).expand(batch_size, -1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch GAN to synthesize aircraft time series."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the OpenSky trajectory CSV.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DataConfig.sequence_length,
        help="Window length for each training sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DataConfig.stride,
        help="Sliding-window stride used to create training samples.",
    )
    parser.add_argument(
        "--min-flight-points",
        type=int,
        default=DataConfig.min_flight_points,
        help="Minimum number of points required to keep a flight.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DataConfig.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=ModelConfig.latent_dim,
        help="Latent noise dimension for the generator.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=ModelConfig.hidden_dim,
        help="Shared hidden width for generator and discriminator.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainConfig.epochs,
        help="Number of GAN training epochs.",
    )
    parser.add_argument(
        "--generator-lr",
        type=float,
        default=TrainConfig.generator_lr,
        help="Learning rate for the generator optimizer.",
    )
    parser.add_argument(
        "--discriminator-lr",
        type=float,
        default=TrainConfig.discriminator_lr,
        help="Learning rate for the discriminator optimizer.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=TrainConfig.print_every,
        help="Epoch interval for progress logging.",
    )
    parser.add_argument(
        "--num-generate",
        type=int,
        default=16,
        help="How many synthetic sequences to generate after training.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("gan_timeseries_sequences.png"),
        help="Optional output path for the qualitative comparison plot.",
    )
    parser.add_argument(
        "--trajectory-overlay-plot-path",
        type=Path,
        default=Path("gan_timeseries_trajectory_overlay.png"),
        help="Output path for the original-vs-synthetic full trajectory overlay.",
    )
    parser.add_argument(
        "--trajectory-envelope-plot-path",
        type=Path,
        default=Path("gan_timeseries_trajectory_envelope.png"),
        help="Output path for the mean trajectory plus 1-sigma envelope plot.",
    )
    parser.add_argument(
        "--raw-data-plot-path",
        type=Path,
        default=Path("gan_timeseries_loaded_altitude.png"),
        help="Output path for the raw altitude-vs-time verification plot.",
    )
    parser.add_argument(
        "--scaled-data-plot-path",
        type=Path,
        default=Path("gan_timeseries_scaled_altitude.png"),
        help="Output path for the scaled altitude-vs-time verification plot.",
    )
    parser.add_argument(
        "--disable-plot",
        action="store_true",
        help="Skip saving the qualitative comparison plot.",
    )
    parser.add_argument(
        "--disable-preprocess-plots",
        action="store_true",
        help="Skip saving the preprocessing verification plots.",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_onground_label(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def build_feature_frame(flight_frame: pd.DataFrame) -> pd.DataFrame:
    ordered_flight = flight_frame.sort_values("time").reset_index(drop=True)
    track_radians = ordered_flight["true_track_deg"].apply(math.radians)
    return pd.DataFrame(
        {
            "elapsed_seconds": (
                ordered_flight["time"] - ordered_flight["time"].iloc[0]
            ).dt.total_seconds(),
            "lat": ordered_flight["lat"],
            "lon": ordered_flight["lon"],
            "altitude_m": ordered_flight["altitude_m"],
            "track_sin": track_radians.apply(math.sin),
            "track_cos": track_radians.apply(math.cos),
        }
    )


def make_feature_tensor(
    flight_frame: pd.DataFrame,
    dtype: torch.dtype,
) -> Tensor:
    feature_frame = build_feature_frame(flight_frame)
    return torch.tensor(feature_frame[list(FEATURE_NAMES)].to_records(index=False).tolist(), dtype=dtype)


def inverse_normalize(batch: Tensor, stats: NormalizationStats) -> Tensor:
    mean = stats.mean.to(device=batch.device, dtype=batch.dtype)
    std = stats.std.to(device=batch.device, dtype=batch.dtype)
    return batch * std + mean


def compute_window_start_indices(
    num_points: int,
    sequence_length: int,
    stride: int,
) -> list[int]:
    if num_points < sequence_length:
        return []

    start_indices = list(range(0, num_points - sequence_length + 1, stride))
    final_start = num_points - sequence_length
    if start_indices[-1] != final_start:
        start_indices.append(final_start)
    return start_indices


def lag_one_autocorrelation(batch: Tensor) -> Tensor:
    if batch.size(1) < 2:
        return torch.zeros(batch.size(-1), device=batch.device, dtype=batch.dtype)

    current_step = batch[:, :-1, :]
    next_step = batch[:, 1:, :]

    current_step = current_step - current_step.mean(dim=(0, 1), keepdim=True)
    next_step = next_step - next_step.mean(dim=(0, 1), keepdim=True)

    numerator = (current_step * next_step).mean(dim=(0, 1))
    denominator = torch.sqrt(
        current_step.square().mean(dim=(0, 1)).clamp_min(1e-8)
        * next_step.square().mean(dim=(0, 1)).clamp_min(1e-8)
    )
    return numerator / denominator


# 1. Load the dataset and preprocess it to create sequences of time series data.

# 1.1 Load and parse the trajectory CSV.
def load_trajectory_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["time", "first_seen", "last_seen"])

    numeric_columns = ["lat", "lon", "altitude_m", "true_track_deg", "duration_hours"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["callsign"] = frame["callsign"].fillna("").astype(str).str.strip()
    frame["origin"] = frame["origin"].fillna("").astype(str).str.strip()
    frame["destination"] = frame["destination"].fillna("").astype(str).str.strip()
    frame["icao24"] = frame["icao24"].fillna("").astype(str).str.strip()
    frame["onground"] = frame["onground"].fillna(False).map(parse_onground_label)

    required_columns = [
        "time",
        "lat",
        "lon",
        "altitude_m",
        "true_track_deg",
        "onground",
        "icao24",
        "callsign",
        "origin",
        "destination",
        "first_seen",
    ]
    frame = frame.dropna(subset=required_columns)
    frame = frame.loc[~frame["onground"]].copy()
    return frame


# 1.2 Group rows into flight-wise sequences ordered by time.
def group_rows_by_flight(frame: pd.DataFrame) -> list[pd.DataFrame]:
    grouped_flights: list[pd.DataFrame] = []
    group_columns = ["icao24", "callsign", "origin", "destination", "first_seen"]

    ordered_frame = frame.sort_values(group_columns + ["time"]).reset_index(drop=True)
    for _, flight_frame in ordered_frame.groupby(group_columns, sort=False, dropna=False):
        grouped_flights.append(flight_frame.reset_index(drop=True).copy())

    return grouped_flights


# 1.3 Convert raw values into torch tensors without NumPy.
def build_flight_tensors(
    grouped_rows: list[pd.DataFrame],
    min_flight_points: int,
    dtype: torch.dtype,
) -> list[Tensor]:
    flight_tensors: list[Tensor] = []

    for flight_frame in grouped_rows:
        if len(flight_frame) < min_flight_points:
            continue
        feature_tensor = make_feature_tensor(flight_frame, dtype=dtype)
        if torch.isfinite(feature_tensor).all():
            flight_tensors.append(feature_tensor)

    return flight_tensors


def filter_valid_flight_frames(
    grouped_rows: list[pd.DataFrame],
    min_flight_points: int,
    dtype: torch.dtype,
) -> list[pd.DataFrame]:
    valid_flights: list[pd.DataFrame] = []
    for flight_frame in grouped_rows:
        if len(flight_frame) < min_flight_points:
            continue
        feature_tensor = make_feature_tensor(flight_frame, dtype=dtype)
        if torch.isfinite(feature_tensor).all():
            valid_flights.append(flight_frame)
    return valid_flights


# 1.4 Normalize features and create sliding windows for GAN training.
def create_training_windows(
    flight_frames: list[pd.DataFrame],
    sequence_length: int,
    stride: int,
    min_flight_points: int,
    dtype: torch.dtype,
) -> tuple[Tensor, NormalizationStats, list[WindowMetadata]]:
    flight_tensors = build_flight_tensors(
        grouped_rows=flight_frames,
        min_flight_points=min_flight_points,
        dtype=dtype,
    )
    if not flight_tensors:
        raise ValueError("No valid flight tensors were created from the CSV.")

    stacked_points = torch.cat(flight_tensors, dim=0)
    mean = stacked_points.mean(dim=0)
    std = stacked_points.std(dim=0, correction=0).clamp_min(1e-6)
    stats = NormalizationStats(mean=mean, std=std)

    windows: list[Tensor] = []
    window_metadata: list[WindowMetadata] = []
    valid_flight_frames = filter_valid_flight_frames(
        grouped_rows=flight_frames,
        min_flight_points=min_flight_points,
        dtype=dtype,
    )

    for flight_index, (flight_tensor, flight_frame) in enumerate(
        zip(flight_tensors, valid_flight_frames, strict=True)
    ):
        normalized_flight = (flight_tensor - mean) / std
        feature_frame = build_feature_frame(flight_frame)
        elapsed_seconds = torch.tensor(
            feature_frame["elapsed_seconds"].tolist(),
            dtype=dtype,
        )
        start_indices = compute_window_start_indices(
            num_points=normalized_flight.size(0),
            sequence_length=sequence_length,
            stride=stride,
        )
        for start_index in start_indices:
            windows.append(normalized_flight[start_index : start_index + sequence_length])
            window_metadata.append(
                WindowMetadata(
                    flight_label=format_flight_label(flight_frame),
                    flight_index=flight_index,
                    start_index=start_index,
                    end_index=start_index + sequence_length,
                    elapsed_seconds=elapsed_seconds[start_index : start_index + sequence_length].clone(),
                )
            )

    if not windows:
        raise ValueError(
            "No training windows were created. Reduce --sequence-length or "
            "--min-flight-points to fit the available trajectory lengths."
        )

    return torch.stack(windows), stats, window_metadata


# 1.5 Validate preprocessing outputs before model construction.
def validate_preprocessing(sequence_batch: Tensor) -> None:
    if sequence_batch.ndim != 3:
        raise ValueError(
            f"Expected window tensor with shape [N, T, F], received {sequence_batch.shape}."
        )
    if sequence_batch.size(0) == 0:
        raise ValueError("No time-series windows are available for GAN training.")
    if not torch.isfinite(sequence_batch).all():
        raise ValueError("Preprocessed training windows contain non-finite values.")


def build_dataloader(sequences: Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def format_flight_label(flight_frame: pd.DataFrame) -> str:
    first_row = flight_frame.iloc[0]
    callsign = str(first_row["callsign"]).strip()
    destination = str(first_row["destination"]).strip()
    return f"{callsign} -> {destination}"


def save_raw_altitude_plot(flight_frames: list[pd.DataFrame], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 6))

    for flight_frame in flight_frames:
        ordered_flight = flight_frame.sort_values("time").reset_index(drop=True)
        hours_since_takeoff = (
            ordered_flight["time"] - ordered_flight["time"].iloc[0]
        ).dt.total_seconds() / 3600.0
        axis.plot(
            hours_since_takeoff,
            ordered_flight["altitude_m"],
            label=format_flight_label(ordered_flight),
        )

    axis.set_xlabel("Hours Since Takeoff")
    axis.set_ylabel("Altitude (m)")
    axis.set_title("Loaded Airborne Flight Data: Altitude vs Time Since Takeoff")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_scaled_altitude_plot(
    flight_frames: list[pd.DataFrame],
    stats: NormalizationStats,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    altitude_index = FEATURE_NAMES.index("altitude_m")
    altitude_mean = float(stats.mean[altitude_index].detach().cpu().item())
    altitude_std = float(stats.std[altitude_index].detach().cpu().item())

    fig, axis = plt.subplots(figsize=(10, 6))

    for flight_frame in flight_frames:
        feature_frame = build_feature_frame(flight_frame)
        hours_since_takeoff = feature_frame["elapsed_seconds"] / 3600.0
        scaled_altitude = (feature_frame["altitude_m"] - altitude_mean) / altitude_std
        axis.plot(
            hours_since_takeoff,
            scaled_altitude,
            label=format_flight_label(flight_frame),
        )

    axis.set_xlabel("Hours Since Takeoff")
    axis.set_ylabel("Scaled Altitude (z-score)")
    axis.set_title("Scaled Airborne Flight Data: Altitude vs Time Since Takeoff")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_real_trajectories(
    flight_frames: list[pd.DataFrame],
    dtype: torch.dtype,
) -> list[FullTrajectory]:
    trajectories: list[FullTrajectory] = []

    for flight_frame in flight_frames:
        feature_frame = build_feature_frame(flight_frame)
        elapsed_seconds = torch.tensor(
            feature_frame["elapsed_seconds"].tolist(),
            dtype=dtype,
        )
        normalized_time = torch.linspace(
            0.0,
            1.0,
            steps=len(feature_frame),
            dtype=dtype,
        )
        feature_tensor = torch.tensor(
            feature_frame[list(FEATURE_NAMES)].to_records(index=False).tolist(),
            dtype=dtype,
        )
        trajectories.append(
            FullTrajectory(
                label=format_flight_label(flight_frame),
                elapsed_seconds=elapsed_seconds,
                normalized_time=normalized_time,
                features=feature_tensor,
            )
        )

    return trajectories


class SyntheticTrajectoryAssembler:
    def __init__(
        self,
        generator: nn.Module,
        normalization_stats: NormalizationStats,
        sequence_length: int,
        stride: int,
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.generator = generator
        self.normalization_stats = normalization_stats
        self.sequence_length = sequence_length
        self.stride = stride
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype

    def _build_overlap_weights(self) -> Tensor:
        if self.sequence_length == 1:
            return torch.ones(1, 1, device=self.device, dtype=self.dtype)

        positions = torch.arange(
            self.sequence_length,
            device=self.device,
            dtype=self.dtype,
        )
        center = (self.sequence_length - 1) / 2.0
        distances = torch.abs((positions - center) / max(center, 1.0))
        weights = (1.0 - 0.5 * distances).clamp_min(0.5)
        return weights.view(-1, 1)

    def synthesize_trajectory(self, reference: FullTrajectory) -> FullTrajectory:
        reference_length = reference.features.size(0)
        start_indices = compute_window_start_indices(
            num_points=reference_length,
            sequence_length=self.sequence_length,
            stride=self.stride,
        )

        accumulated = torch.zeros(
            reference_length,
            reference.features.size(1),
            device=self.device,
            dtype=self.dtype,
        )
        accumulated_weights = torch.zeros(
            reference_length,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        window_weights = self._build_overlap_weights()

        self.generator.eval()
        with torch.no_grad():
            for start_index in start_indices:
                noise = torch.randn(
                    1,
                    self.latent_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                time_channel = build_normalized_time_channel(
                    batch_size=1,
                    sequence_length=self.sequence_length,
                    device=self.device,
                    dtype=self.dtype,
                )
                generated_window = self.generator(noise, time_channel)
                generated_window = inverse_normalize(
                    generated_window,
                    self.normalization_stats,
                )[0]
                accumulated[start_index : start_index + self.sequence_length] += (
                    generated_window * window_weights
                )
                accumulated_weights[
                    start_index : start_index + self.sequence_length
                ] += window_weights

        full_features = accumulated / accumulated_weights.clamp_min(1e-6)
        return FullTrajectory(
            label=f"Synthetic {reference.label}",
            elapsed_seconds=reference.elapsed_seconds.detach().cpu().clone(),
            normalized_time=reference.normalized_time.detach().cpu().clone(),
            features=full_features.detach().cpu(),
        )

    def synthesize_all(
        self,
        references: list[FullTrajectory],
    ) -> list[FullTrajectory]:
        return [self.synthesize_trajectory(reference) for reference in references]


def save_trajectory_overlay_plot(
    real_trajectories: list[FullTrajectory],
    synthetic_trajectories: list[FullTrajectory],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    altitude_index = FEATURE_NAMES.index("altitude_m")
    fig, axis = plt.subplots(figsize=(10, 6))

    for real_trajectory, synthetic_trajectory in zip(
        real_trajectories,
        synthetic_trajectories,
        strict=True,
    ):
        elapsed_hours = (real_trajectory.elapsed_seconds / 3600.0).tolist()
        real_altitude = real_trajectory.features[:, altitude_index].tolist()
        synthetic_altitude = synthetic_trajectory.features[:, altitude_index].tolist()

        real_line = axis.plot(
            elapsed_hours,
            real_altitude,
            linewidth=1.8,
            label=f"Real {real_trajectory.label}",
        )[0]
        axis.plot(
            elapsed_hours,
            synthetic_altitude,
            linestyle="--",
            linewidth=1.5,
            color=real_line.get_color(),
            label=f"Synthetic {real_trajectory.label}",
        )

    axis.set_xlabel("Hours Since Takeoff")
    axis.set_ylabel("Altitude (m)")
    axis.set_title("Original and Synthetic Full Trajectories")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def resample_trajectory_feature(
    trajectories: list[FullTrajectory],
    feature_name: str,
    num_points: int,
) -> tuple[Tensor, Tensor]:
    feature_index = FEATURE_NAMES.index(feature_name)
    resampled_series: list[Tensor] = []

    for trajectory in trajectories:
        feature_series = trajectory.features[:, feature_index].view(1, 1, -1)
        resized_series = F.interpolate(
            feature_series,
            size=num_points,
            mode="linear",
            align_corners=True,
        )
        resampled_series.append(resized_series[0, 0])

    stacked_series = torch.stack(resampled_series)
    normalized_axis = torch.linspace(0.0, 1.0, steps=num_points, dtype=stacked_series.dtype)
    return normalized_axis, stacked_series


def save_trajectory_envelope_plot(
    real_trajectories: list[FullTrajectory],
    synthetic_trajectories: list[FullTrajectory],
    output_path: Path,
    num_points: int = 128,
) -> None:
    import matplotlib.pyplot as plt

    normalized_axis, real_altitudes = resample_trajectory_feature(
        trajectories=real_trajectories,
        feature_name="altitude_m",
        num_points=num_points,
    )
    _, synthetic_altitudes = resample_trajectory_feature(
        trajectories=synthetic_trajectories,
        feature_name="altitude_m",
        num_points=num_points,
    )

    real_mean = real_altitudes.mean(dim=0)
    real_std = real_altitudes.std(dim=0, correction=0)
    synthetic_mean = synthetic_altitudes.mean(dim=0)
    synthetic_std = synthetic_altitudes.std(dim=0, correction=0)

    fig, axis = plt.subplots(figsize=(10, 6))

    axis.fill_between(
        normalized_axis.tolist(),
        (real_mean - real_std).tolist(),
        (real_mean + real_std).tolist(),
        alpha=0.2,
        label="Real +/- 1 sigma",
    )
    axis.plot(
        normalized_axis.tolist(),
        real_mean.tolist(),
        linewidth=2.0,
        label="Real mean altitude",
    )
    axis.fill_between(
        normalized_axis.tolist(),
        (synthetic_mean - synthetic_std).tolist(),
        (synthetic_mean + synthetic_std).tolist(),
        alpha=0.2,
        label="Synthetic +/- 1 sigma",
    )
    axis.plot(
        normalized_axis.tolist(),
        synthetic_mean.tolist(),
        linewidth=2.0,
        linestyle="--",
        label="Synthetic mean altitude",
    )

    axis.set_xlabel("Normalized Time")
    axis.set_ylabel("Altitude (m)")
    axis.set_title("Mean Altitude Trajectory with 1-Sigma Envelopes")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# 2. Define a GAN architecture suitable for time series data.

# 2.1 Generator: convert the article's dense generator to PyTorch and adapt it
#     to emit full time-series windows instead of single steps.
class TimeSeriesGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        sequence_length: int,
        feature_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.noise_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length * hidden_dim),
            nn.ReLU(),
        )
        self.sequence_model = nn.GRU(
            input_size=hidden_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, noise: Tensor, time_channel: Tensor) -> Tensor:
        hidden_sequence = self.noise_projector(noise)
        hidden_sequence = hidden_sequence.view(
            noise.size(0), self.sequence_length, self.hidden_dim
        )
        conditioned_sequence = torch.cat([hidden_sequence, time_channel], dim=-1)
        generated_sequence, _ = self.sequence_model(conditioned_sequence)
        return self.output_layer(generated_sequence)


# 2.2 Discriminator: convert the article's dense discriminator to PyTorch and
#     adapt it to score whole time-series windows.
class TimeSeriesDiscriminator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.sequence_model = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence: Tensor, time_channel: Tensor) -> Tensor:
        conditioned_sequence = torch.cat([sequence, time_channel], dim=-1)
        encoded_sequence = self.feature_encoder(conditioned_sequence)
        _, hidden_state = self.sequence_model(encoded_sequence)
        return self.classifier(hidden_state[-1]).squeeze(-1)


# 2.3 Assemble the GAN training components and validate tensor shapes.
def build_gan_components(
    feature_dim: int,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> GANComponents:
    generator = TimeSeriesGenerator(
        latent_dim=model_config.latent_dim,
        hidden_dim=model_config.hidden_dim,
        sequence_length=data_config.sequence_length,
        feature_dim=feature_dim,
        num_layers=model_config.num_layers,
    ).to(device=device, dtype=dtype)

    discriminator = TimeSeriesDiscriminator(
        feature_dim=feature_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
    ).to(device=device, dtype=dtype)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=train_config.generator_lr,
        betas=(train_config.beta1, train_config.beta2),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_config.discriminator_lr,
        betas=(train_config.beta1, train_config.beta2),
    )
    criterion = nn.BCEWithLogitsLoss()

    return GANComponents(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        criterion=criterion,
    )


def validate_models(
    components: GANComponents,
    sample_batch: Tensor,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    batch_size = min(sample_batch.size(0), 4)
    sample_batch = sample_batch[:batch_size].to(device=device, dtype=dtype)
    noise = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    time_channel = build_normalized_time_channel(
        batch_size=batch_size,
        sequence_length=sample_batch.size(1),
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        generated_batch = components.generator(noise, time_channel)
        real_logits = components.discriminator(sample_batch, time_channel)
        fake_logits = components.discriminator(generated_batch, time_channel)

    if generated_batch.shape != sample_batch.shape:
        raise ValueError(
            "Generator output shape does not match the training batch shape: "
            f"{generated_batch.shape} vs {sample_batch.shape}."
        )
    if real_logits.shape != torch.Size([batch_size]):
        raise ValueError(f"Unexpected discriminator output shape: {real_logits.shape}.")
    if fake_logits.shape != torch.Size([batch_size]):
        raise ValueError(
            "Unexpected discriminator output shape on generated data: "
            f"{fake_logits.shape}."
        )


# 3. Train the GAN on the preprocessed dataset.

# 3.1 Train the discriminator on real and generated batches.
def train_discriminator_step(
    components: GANComponents,
    real_batch: Tensor,
    latent_dim: int,
    label_smoothing: float,
) -> float:
    batch_size = real_batch.size(0)
    device = real_batch.device
    dtype = real_batch.dtype

    real_targets = torch.full(
        (batch_size,),
        fill_value=label_smoothing,
        device=device,
        dtype=dtype,
    )
    fake_targets = torch.zeros(batch_size, device=device, dtype=dtype)
    noise = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    time_channel = build_normalized_time_channel(
        batch_size=batch_size,
        sequence_length=real_batch.size(1),
        device=device,
        dtype=dtype,
    )
    fake_batch = components.generator(noise, time_channel).detach()

    components.discriminator_optimizer.zero_grad(set_to_none=True)
    real_logits = components.discriminator(real_batch, time_channel)
    fake_logits = components.discriminator(fake_batch, time_channel)
    real_loss = components.criterion(real_logits, real_targets)
    fake_loss = components.criterion(fake_logits, fake_targets)
    discriminator_loss = real_loss + fake_loss
    discriminator_loss.backward()
    components.discriminator_optimizer.step()

    return float(discriminator_loss.detach().cpu().item())


# 3.2 Train the generator to fool the discriminator.
def train_generator_step(
    components: GANComponents,
    batch_size: int,
    latent_dim: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    target_is_real = torch.ones(batch_size, device=device, dtype=dtype)
    noise = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    time_channel = build_normalized_time_channel(
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
        dtype=dtype,
    )

    components.generator_optimizer.zero_grad(set_to_none=True)
    generated_batch = components.generator(noise, time_channel)
    generated_logits = components.discriminator(generated_batch, time_channel)
    generator_loss = components.criterion(generated_logits, target_is_real)
    generator_loss.backward()
    components.generator_optimizer.step()

    return float(generator_loss.detach().cpu().item())


# 3.3 Run the adversarial loop and validate the loss history.
def train_gan(
    components: GANComponents,
    dataloader: DataLoader,
    train_config: TrainConfig,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, list[float]]:
    history = {"generator_loss": [], "discriminator_loss": []}

    for epoch in range(train_config.epochs):
        epoch_generator_loss = 0.0
        epoch_discriminator_loss = 0.0
        num_batches = 0

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device=device, dtype=dtype)
            discriminator_loss = train_discriminator_step(
                components=components,
                real_batch=real_batch,
                latent_dim=latent_dim,
                label_smoothing=train_config.label_smoothing,
            )
            generator_loss = train_generator_step(
                components=components,
                batch_size=real_batch.size(0),
                latent_dim=latent_dim,
                sequence_length=real_batch.size(1),
                device=device,
                dtype=dtype,
            )

            epoch_discriminator_loss += discriminator_loss
            epoch_generator_loss += generator_loss
            num_batches += 1

        mean_discriminator_loss = epoch_discriminator_loss / max(num_batches, 1)
        mean_generator_loss = epoch_generator_loss / max(num_batches, 1)
        history["discriminator_loss"].append(mean_discriminator_loss)
        history["generator_loss"].append(mean_generator_loss)

        if epoch == 0 or (epoch + 1) % train_config.print_every == 0:
            print(
                f"Epoch {epoch + 1:04d}/{train_config.epochs} | "
                f"D loss: {mean_discriminator_loss:.4f} | "
                f"G loss: {mean_generator_loss:.4f}"
            )

    return history


def validate_training_history(history: dict[str, list[float]]) -> None:
    for key, values in history.items():
        if not values:
            raise ValueError(f"Training history for {key} is empty.")
        loss_tensor = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(loss_tensor).all():
            raise ValueError(f"Training history for {key} contains non-finite values.")


# 4. Generate synthetic time series data using the trained GAN and evaluate its quality.

# 4.1 Sample synthetic sequences from the trained generator.
def generate_synthetic_sequences(
    generator: nn.Module,
    num_sequences: int,
    latent_dim: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_sequences, latent_dim, device=device, dtype=dtype)
        time_channel = build_normalized_time_channel(
            batch_size=num_sequences,
            sequence_length=sequence_length,
            device=device,
            dtype=dtype,
        )
        synthetic_sequences = generator(noise, time_channel)
    generator.train()
    return synthetic_sequences


# 4.2 Invert normalization back to physical units.
def sample_real_sequences(
    sequences: Tensor,
    window_metadata: list[WindowMetadata],
    num_sequences: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, list[WindowMetadata]]:
    sample_count = min(num_sequences, sequences.size(0))
    indices = torch.randperm(sequences.size(0))[:sample_count]
    sampled_metadata = [window_metadata[index] for index in indices.tolist()]
    return sequences[indices].to(device=device, dtype=dtype), sampled_metadata


# 4.3 Evaluate distribution, trajectory, and temporal quality metrics.
def evaluate_synthetic_quality(real_batch: Tensor, synthetic_batch: Tensor) -> dict[str, Tensor]:
    real_feature_mean = real_batch.mean(dim=(0, 1))
    synthetic_feature_mean = synthetic_batch.mean(dim=(0, 1))
    real_feature_std = real_batch.std(dim=(0, 1), correction=0)
    synthetic_feature_std = synthetic_batch.std(dim=(0, 1), correction=0)

    mean_absolute_gap = (synthetic_feature_mean - real_feature_mean).abs()
    std_absolute_gap = (synthetic_feature_std - real_feature_std).abs()

    real_mean_trajectory = real_batch.mean(dim=0)
    synthetic_mean_trajectory = synthetic_batch.mean(dim=0)
    trajectory_rmse = torch.sqrt(
        (synthetic_mean_trajectory - real_mean_trajectory).square().mean()
    )

    real_autocorr = lag_one_autocorrelation(real_batch)
    synthetic_autocorr = lag_one_autocorrelation(synthetic_batch)
    autocorr_gap = (synthetic_autocorr - real_autocorr).abs()

    return {
        "feature_mean_gap": mean_absolute_gap,
        "feature_std_gap": std_absolute_gap,
        "trajectory_rmse": trajectory_rmse,
        "real_autocorr": real_autocorr,
        "synthetic_autocorr": synthetic_autocorr,
        "autocorr_gap": autocorr_gap,
    }


def print_evaluation(metrics: dict[str, Tensor]) -> None:
    mean_gap = metrics["feature_mean_gap"].detach().cpu()
    std_gap = metrics["feature_std_gap"].detach().cpu()
    autocorr_gap = metrics["autocorr_gap"].detach().cpu()
    trajectory_rmse = float(metrics["trajectory_rmse"].detach().cpu().item())

    print("\nEvaluation summary")
    print(f"Trajectory RMSE (mean path proxy): {trajectory_rmse:.4f}")
    for feature_name, feature_mean_gap, feature_std_gap, feature_autocorr_gap in zip(
        FEATURE_NAMES,
        mean_gap.tolist(),
        std_gap.tolist(),
        autocorr_gap.tolist(),
    ):
        print(
            f"{feature_name:>16s} | "
            f"mean gap: {feature_mean_gap:9.4f} | "
            f"std gap: {feature_std_gap:9.4f} | "
            f"lag-1 autocorr gap: {feature_autocorr_gap:9.4f}"
        )


# 4.4 Optionally save comparison plots for qualitative inspection.
def save_comparison_plot(
    real_batch: Tensor,
    synthetic_batch: Tensor,
    real_window_metadata: list[WindowMetadata],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    altitude_index = FEATURE_NAMES.index("altitude_m")
    real_sample = real_batch[0].detach().cpu()
    synthetic_sample = synthetic_batch[0].detach().cpu()
    step_axis = torch.arange(real_sample.size(0), dtype=torch.int64).tolist()
    normalized_time_axis = build_normalized_time_channel(
        batch_size=1,
        sequence_length=real_sample.size(0),
        device=torch.device("cpu"),
        dtype=real_sample.dtype,
    )[0, :, 0].tolist()

    elapsed_time_axis = real_window_metadata[0].elapsed_seconds.detach().cpu().tolist()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(
        elapsed_time_axis,
        real_sample[:, altitude_index].tolist(),
        label="Real",
    )
    axes[0].plot(
        elapsed_time_axis,
        synthetic_sample[:, altitude_index].tolist(),
        label="Synthetic",
    )
    axes[0].set_title("Altitude vs elapsed time")
    axes[0].set_xlabel("Elapsed seconds")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        normalized_time_axis,
        real_sample[:, altitude_index].tolist(),
        label="Real",
    )
    axes[1].plot(
        normalized_time_axis,
        synthetic_sample[:, altitude_index].tolist(),
        label="Synthetic",
    )
    axes[1].set_title("Altitude vs normalized time")
    axes[1].set_xlabel("Normalized time")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    real_mean_altitude = real_batch[:, :, altitude_index].mean(dim=0).detach().cpu()
    synthetic_mean_altitude = (
        synthetic_batch[:, :, altitude_index].mean(dim=0).detach().cpu()
    )
    axes[2].plot(step_axis, real_mean_altitude.tolist(), label="Real mean altitude")
    axes[2].plot(
        step_axis,
        synthetic_mean_altitude.tolist(),
        label="Synthetic mean altitude",
    )
    axes[2].set_title("Mean altitude profile")
    axes[2].set_xlabel("Sequence step")
    axes[2].set_ylabel("Altitude (m)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = select_device()
    dtype = torch.float32

    data_config = DataConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        min_flight_points=args.min_flight_points,
        batch_size=args.batch_size,
    )
    model_config = ModelConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        print_every=args.print_every,
    )

    print(f"Using device: {device}")
    print(f"Reading trajectories from: {args.data_path}")

    frame = load_trajectory_frame(args.data_path)
    grouped_rows = group_rows_by_flight(frame)
    valid_flight_frames = filter_valid_flight_frames(
        grouped_rows=grouped_rows,
        min_flight_points=data_config.min_flight_points,
        dtype=dtype,
    )
    sequences, normalization_stats, window_metadata = create_training_windows(
        flight_frames=valid_flight_frames,
        sequence_length=data_config.sequence_length,
        stride=data_config.stride,
        min_flight_points=data_config.min_flight_points,
        dtype=dtype,
    )
    validate_preprocessing(sequences)

    if not args.disable_preprocess_plots:
        save_raw_altitude_plot(
            flight_frames=valid_flight_frames,
            output_path=args.raw_data_plot_path,
        )
        save_scaled_altitude_plot(
            flight_frames=valid_flight_frames,
            stats=normalization_stats,
            output_path=args.scaled_data_plot_path,
        )

    print(
        "Preprocessing complete | "
        f"flights kept: {len(valid_flight_frames)} | "
        f"windows: {sequences.size(0)} | "
        f"window shape: {tuple(sequences.shape[1:])}"
    )
    if not args.disable_preprocess_plots:
        print(f"Saved raw data plot to: {args.raw_data_plot_path}")
        print(f"Saved scaled data plot to: {args.scaled_data_plot_path}")

    dataloader = build_dataloader(sequences, batch_size=data_config.batch_size)
    components = build_gan_components(
        feature_dim=sequences.size(-1),
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        device=device,
        dtype=dtype,
    )
    validate_models(
        components=components,
        sample_batch=sequences,
        latent_dim=model_config.latent_dim,
        device=device,
        dtype=dtype,
    )
    print("Model validation complete.")

    history = train_gan(
        components=components,
        dataloader=dataloader,
        train_config=train_config,
        latent_dim=model_config.latent_dim,
        device=device,
        dtype=dtype,
    )
    validate_training_history(history)
    print("Training validation complete.")

    synthetic_batch_normalized = generate_synthetic_sequences(
        generator=components.generator,
        num_sequences=args.num_generate,
        latent_dim=model_config.latent_dim,
        sequence_length=data_config.sequence_length,
        device=device,
        dtype=dtype,
    )
    real_batch_normalized, sampled_real_metadata = sample_real_sequences(
        sequences=sequences,
        window_metadata=window_metadata,
        num_sequences=args.num_generate,
        device=device,
        dtype=dtype,
    )

    synthetic_batch = inverse_normalize(synthetic_batch_normalized, normalization_stats)
    real_batch = inverse_normalize(real_batch_normalized, normalization_stats)

    metrics = evaluate_synthetic_quality(
        real_batch=real_batch,
        synthetic_batch=synthetic_batch,
    )
    print_evaluation(metrics)

    real_trajectories = build_real_trajectories(
        flight_frames=valid_flight_frames,
        dtype=dtype,
    )
    trajectory_assembler = SyntheticTrajectoryAssembler(
        generator=components.generator,
        normalization_stats=normalization_stats,
        sequence_length=data_config.sequence_length,
        stride=data_config.stride,
        latent_dim=model_config.latent_dim,
        device=device,
        dtype=dtype,
    )
    synthetic_trajectories = trajectory_assembler.synthesize_all(real_trajectories)

    if not args.disable_plot:
        save_comparison_plot(
            real_batch=real_batch,
            synthetic_batch=synthetic_batch,
            real_window_metadata=sampled_real_metadata,
            output_path=args.plot_path,
        )
        save_trajectory_overlay_plot(
            real_trajectories=real_trajectories,
            synthetic_trajectories=synthetic_trajectories,
            output_path=args.trajectory_overlay_plot_path,
        )
        save_trajectory_envelope_plot(
            real_trajectories=real_trajectories,
            synthetic_trajectories=synthetic_trajectories,
            output_path=args.trajectory_envelope_plot_path,
        )
        print(f"Saved comparison plot to: {args.plot_path}")
        print(f"Saved trajectory overlay plot to: {args.trajectory_overlay_plot_path}")
        print(f"Saved trajectory envelope plot to: {args.trajectory_envelope_plot_path}")


if __name__ == "__main__":
    main()
