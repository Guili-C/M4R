"""
Simplified FNO script for tracking spectral weight evolution during training
(Using Custom FNO definition with Combined Weight Visualization for Multiple Layers)
====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm, Normalize
from pathlib import Path # Use pathlib for cleaner path handling
import time # For timing

# Adjust import path if your neuralop installation differs
try:
    from neuraloperator.neuralop.data.datasets import DarcyDataset
except ImportError:
    print("Warning: neuraloperator package not found or DarcyDataset unavailable.")

# --- Configuration ---
OUTPUT_DIR = Path('./fno_spectral_analysis_results')
N_TRAIN = 1000
BATCH_SIZE = 32
TRAIN_RESOLUTION = 64
TEST_RESOLUTION = 64
N_TEST = 100
EPOCHS = 40
SAVE_FREQ = 5 # Frequency (in epochs) to save weight snapshots
LEARNING_RATE = 8e-3
WEIGHT_DECAY = 1e-4
MODES = 32 # Modes for both dimensions
WIDTH = 32 # FNO width (hidden channels)
N_VIS_SAMPLES = 3 # Number of prediction samples to visualize
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --- End Configuration ---

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "weights").mkdir(exist_ok=True)
(OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# =====================================================================
# Custom FNO Model Definition
# =====================================================================

class SpectralConv2d(nn.Module):
    """ 2D Fourier layer using rfft for real inputs. """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm='ortho') # Use ortho norm is common

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Multiply relevant Fourier modes
        out_ft[..., :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[..., :self.modes1, :self.modes2], self.weights1)
        out_ft[..., -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[..., -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # Layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # Layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# =====================================================================
# Helper Functions
# =====================================================================

def get_grid2d(shape, device):
    """Generates a 2D mesh grid normalized to [0,1]"""
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    grid_x = torch.linspace(0, 1, size_x, device=device, dtype=torch.float32)
    grid_y = torch.linspace(0, 1, size_y, device=device, dtype=torch.float32)
    mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    mesh_x = mesh_x.unsqueeze(0).unsqueeze(-1).expand(batchsize, -1, -1, 1)
    mesh_y = mesh_y.unsqueeze(0).unsqueeze(-1).expand(batchsize, -1, -1, 1)
    return mesh_x, mesh_y

def adapt_input_data(x, device):
    """ Adds coordinate channels (x, y) to the input data tensor. """
    x_permuted = x.permute(0, 2, 3, 1)
    mesh_x, mesh_y = get_grid2d(x_permuted.shape, device)
    x_adapted = torch.cat((x_permuted, mesh_x, mesh_y), dim=-1)
    return x_adapted

def save_figure(fig, path):
    """Helper to save and close a matplotlib figure."""
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_training_metrics(metrics, epochs, output_path):
    """Plots training loss and learning rate."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, epochs + 1)

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss', color=color)
    ax1.plot(epochs_range, metrics['train_losses'], 'o-', color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(epochs_range, metrics['learning_rates'], 's--', color=color, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Training Loss and Learning Rate Evolution')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, output_path)

# =====================================================================
# Training Function
# =====================================================================

def train_model(model, train_loader, epochs, save_freq, lr, weight_decay, output_dir):
    """Trains the model and tracks spectral weight evolution for ALL SpectralConv2d layers."""
    
    # Dynamically find all spectral weights (weights1 and weights2 from SpectralConv2d layers)
    spectral_weight_names = []
    for name, module in model.named_modules():
        if isinstance(module, SpectralConv2d):
            # Ensure the parameters exist before adding their names
            if hasattr(module, 'weights1'):
                spectral_weight_names.append(f"{name}.weights1")
            if hasattr(module, 'weights2'):
                spectral_weight_names.append(f"{name}.weights2")
    
    # Get the actual parameter objects for these names
    spectral_weights_params = {name: param for name, param in model.named_parameters()
                               if name in spectral_weight_names}

    if not spectral_weights_params:
        print("Warning: No spectral weights (e.g., 'conv0.weights1') found in the model. Skipping weight evolution tracking.")
        weight_evolution = {}
    else:
        print(f"Tracking spectral weights for: {list(spectral_weights_params.keys())}")
        weight_evolution = {name: [p.detach().cpu().clone()] for name, p in spectral_weights_params.items()}
    
    prev_weights = {name: p.detach().cpu().clone() for name, p in spectral_weights_params.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    training_metrics = {'train_losses': [], 'learning_rates': []}
    start_time = time.time()

    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_raw, y_raw = batch['x'].to(DEVICE), batch['y'].to(DEVICE)

            x = adapt_input_data(x_raw, DEVICE)
            y = y_raw.permute(0, 2, 3, 1)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        training_metrics['train_losses'].append(avg_loss)
        training_metrics['learning_rates'].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}", end="")

        if spectral_weights_params and ((epoch + 1) % save_freq == 0 or (epoch + 1) == epochs):
            total_change = 0.0
            # Retrieve current weights for all tracked spectral layers
            current_spectral_weights_values = {name: p for name, p in model.named_parameters() 
                                               if name in spectral_weights_params}
            
            for name, param in current_spectral_weights_values.items():
                param_cpu = param.detach().cpu().clone()
                weight_evolution[name].append(param_cpu)
                change = torch.abs(param_cpu - prev_weights[name]).mean().item()
                total_change += change
                prev_weights[name] = param_cpu
            avg_change = total_change / len(spectral_weights_params) if spectral_weights_params else 0
            print(f" | Weights saved. Avg change: {avg_change:.6f}")
        else:
            print()

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    plot_training_metrics(training_metrics, epochs, output_dir / "predictions/training_metrics.png")

    return weight_evolution, model, save_freq

# =====================================================================
# Visualization Functions (Combined Weights + Predictions)
# =====================================================================

def _calculate_combined_weights(weight_evolution, weights1_key, weights2_key):
    """Helper to calculate stacked, averaged weight magnitudes for a specific layer."""
    if weights1_key not in weight_evolution or weights2_key not in weight_evolution:
        # This case should be handled by the caller, but as a safeguard:
        raise ValueError(f"Missing '{weights1_key}' or '{weights2_key}' in weight_evolution for calculation.")

    snapshots1 = weight_evolution[weights1_key]
    snapshots2 = weight_evolution[weights2_key]
    
    if not snapshots1 or not snapshots2: # Check if lists are empty
        print(f"Warning: Empty snapshot list for {weights1_key} or {weights2_key}.")
        return [], None, None

    if len(snapshots1) != len(snapshots2):
        print(f"Warning: Mismatched snapshots for {weights1_key} ({len(snapshots1)}) and {weights2_key} ({len(snapshots2)}). Using min length.")
        num_snapshots = min(len(snapshots1), len(snapshots2))
        snapshots1, snapshots2 = snapshots1[:num_snapshots], snapshots2[:num_snapshots]
    else:
        num_snapshots = len(snapshots1)

    if num_snapshots == 0: return [], None, None

    shape1 = snapshots1[0].shape
    if shape1 != snapshots2[0].shape:
        raise ValueError(f"Shapes of {weights1_key} {shape1} and {weights2_key} {snapshots2[0].shape} mismatch.")

    modes1, modes2 = shape1[-2], shape1[-1]
    # For in_channels, out_channels, modes1, modes2 (4D tensor)
    # or modes1, modes2 (2D tensor if in_channels/out_channels are 1 and squeezed)
    # The original code assumes weights are [in_c, out_c, modes1, modes2]
    # Mean is taken over in_channels and out_channels
    channel_dims = tuple(range(len(shape1) - 2)) # Dims to average over, typically (0, 1)

    combined_mode_averaged_weights = []
    for i in range(num_snapshots):
        avg_mag1 = torch.abs(snapshots1[i]).mean(dim=channel_dims).numpy()
        avg_mag2 = torch.abs(snapshots2[i]).mean(dim=channel_dims).numpy()
        combined_avg_mag = np.vstack((avg_mag2, avg_mag1)) # [2*modes1, modes2]
        combined_mode_averaged_weights.append(combined_avg_mag)

    return combined_mode_averaged_weights, modes1, modes2


def visualize_combined_weight_evolution(weight_evolution, save_path_base, save_freq):
    """ 
    Visualize combined evolution of weights for ALL SpectralConv2d layers.
    INCLUDES: 
    1. Heatmaps of magnitudes (local color scale).
    2. Heatmaps of change from initial weights.
    3. Plots of 1D mode profiles (evolution and final).
    """
    if not weight_evolution:
        print("No weight evolution data to visualize.")
        return

    # Identify unique spectral layer prefixes
    spectral_layer_prefixes = sorted(list(set([
        name.split('.')[0] for name in weight_evolution.keys()
        if 'weights1' in name or 'weights2' in name
    ])))

    if not spectral_layer_prefixes:
        print("Could not identify any spectral layers from weight_evolution keys.")
        return

    print(f"\nStarting visualization for spectral layers: {spectral_layer_prefixes}")

    for layer_prefix in spectral_layer_prefixes:
        print(f"\n--- Visualizing for Layer: {layer_prefix} ---")
        weights1_key = f"{layer_prefix}.weights1"
        weights2_key = f"{layer_prefix}.weights2"

        if weights1_key not in weight_evolution or weights2_key not in weight_evolution:
            print(f"Skipping visualization for {layer_prefix}: missing keys.")
            continue
        
        save_path = Path(save_path_base)

        try:
            combined_weights, modes1, modes2 = _calculate_combined_weights(
                weight_evolution, weights1_key, weights2_key
            )
        except ValueError as e:
            print(f"Error preparing combined weights for {layer_prefix}: {e}")
            continue
        
        if not combined_weights:
            print(f"No snapshots found for visualization for layer {layer_prefix}.")
            continue

        num_snapshots = len(combined_weights)
        combined_modes1 = 2 * modes1
        plot_file_prefix = f"{layer_prefix}_weights_combined"

        print(f"Visualizing combined frequency mode evolution for {plot_file_prefix}")
        
        # --- Visualize magnitude ---
        max_cols = 5
        cols = min(max_cols, num_snapshots)
        rows = (num_snapshots + cols - 1) // cols
        fig_mag, axes_mag = plt.subplots(rows, cols, figsize=(15, max(5, 3 * rows)), squeeze=False)
        fig_mag.suptitle(f"Layer {layer_prefix}: Combined Mode-Averaged Weight Magnitudes (Local Log Scale)", fontsize=16)

        for i, ax in enumerate(axes_mag.flat):
            if i < num_snapshots:
                epoch_num = (i * save_freq) if i > 0 else 0
                
                current_snapshot_weights = combined_weights[i]
                valid_vals = current_snapshot_weights[current_snapshot_weights > 1e-12]

                if len(valid_vals) > 0:
                    vmin = valid_vals.min()
                    vmax = valid_vals.max()
                    if vmin >= vmax:
                         vmax = vmin + 1e-9
                    local_log_norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    local_log_norm = LogNorm(vmin=1e-10, vmax=1.0)
                
                im = ax.imshow(current_snapshot_weights, cmap='viridis', norm=local_log_norm, aspect='auto')
                
                ax.set_title(f'Epoch {epoch_num}')
                ax.set_xlabel(f'Mode j ({modes2})')
                ax.set_ylabel(f'Stacked i ({combined_modes1})')
                ax.axhline(y=modes1 - 0.5, color='red', linestyle='--', linewidth=0.8)
                fig_mag.colorbar(im, ax=ax, label='Avg Mag', fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig_mag, save_path / f"{plot_file_prefix}_magnitudes_local_scale.png")

        # --- Visualize change from initial weights ---
        if num_snapshots > 1:
            epochs_axis = [0] + list(range(save_freq, (num_snapshots - 1) * save_freq + 1, save_freq))

            initial_weights = combined_weights[0]
            changes_from_initial = [combined_weights[i] - initial_weights for i in range(1, num_snapshots)]
            
            all_changes = np.concatenate([c.flatten() for c in changes_from_initial])
            change_max_abs = np.max(np.abs(all_changes)) if len(all_changes) > 0 else 1.0
            change_max_abs = max(change_max_abs, 1e-9)
            change_norm = Normalize(vmin=-change_max_abs, vmax=change_max_abs)

            rows_chg = (num_snapshots - 1 + cols - 1) // cols
            fig_chg, axes_chg = plt.subplots(rows_chg, cols, figsize=(15, max(5, 3 * rows_chg)), squeeze=False)
            fig_chg.suptitle(f"Layer {layer_prefix}: Change From Initial Weights", fontsize=16)

            for i, ax in enumerate(axes_chg.flat):
                if i < len(changes_from_initial):
                    end_epoch = epochs_axis[i + 1]
                    im = ax.imshow(changes_from_initial[i], cmap='RdBu_r', norm=change_norm, aspect='auto')
                    ax.set_title(f'Epoch 0 → {end_epoch}')
                    ax.set_xlabel(f'Mode j ({modes2})')
                    ax.set_ylabel(f'Stacked i ({combined_modes1})')
                    ax.axhline(y=modes1 - 0.5, color='black', linestyle='--', linewidth=0.8)
                    fig_chg.colorbar(im, ax=ax, label='Change', fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(fig_chg, save_path / f"{plot_file_prefix}_change_from_initial.png")


            # --- Evolution of mode profiles ---
            num_lines = min(num_snapshots, 6)
            selected_indices = np.linspace(0, num_snapshots - 1, num_lines, dtype=int)
            colors_prof = plt.cm.coolwarm(np.linspace(0, 1, len(selected_indices)))

            fig_prof_evo, (ax_i, ax_j) = plt.subplots(1, 2, figsize=(15, 6))
            fig_prof_evo.suptitle(f"Layer {layer_prefix}: Evolution of Combined Mode Magnitude Profiles (Log Scale)")

            for k, snapshot_idx in enumerate(selected_indices):
                epoch_num = epochs_axis[snapshot_idx]
                label = f'Epoch {epoch_num}'
                profile = combined_weights[snapshot_idx]
                ax_i.plot(np.arange(profile.shape[0]), np.mean(profile, axis=1), '-', label=label, color=colors_prof[k])
                ax_j.plot(np.arange(profile.shape[1]), np.mean(profile, axis=0), '-', label=label, color=colors_prof[k])

            ax_i.set_title('Stacked i-Mode Profiles')
            ax_i.set_xlabel(f'Stacked Mode i Index (0 to {combined_modes1-1})')
            ax_i.set_ylabel('Avg Magnitude')
            ax_i.axvline(x=modes1 - 0.5, color='red', linestyle='--', linewidth=0.8)
            ax_i.set_yscale('log')
            ax_i.grid(True, alpha=0.3)
            ax_i.legend(fontsize='small')

            ax_j.set_title('j-Mode Profiles')
            ax_j.set_xlabel(f'Mode j Index (0 to {modes2-1})')
            ax_j.set_ylabel('Avg Magnitude')
            ax_j.set_yscale('log')
            ax_j.grid(True, alpha=0.3)
            ax_j.legend(fontsize='small')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(fig_prof_evo, save_path / f"{plot_file_prefix}_profile_evolution.png")


        # --- Final mode profiles ---
        fig_prof, (ax_i, ax_j) = plt.subplots(1, 2, figsize=(15, 6))
        fig_prof.suptitle(f"Layer {layer_prefix}: Final Combined Mode Magnitude Profiles (Log Scale)")
        final_profile = combined_weights[-1]

        ax_i.plot(np.arange(final_profile.shape[0]), np.mean(final_profile, axis=1), 'o-')
        ax_i.set_title('Stacked i-Mode Profile (Final)')
        ax_i.set_xlabel(f'Stacked Mode i Index (0 to {combined_modes1-1})')
        ax_i.set_ylabel('Avg Magnitude')
        ax_i.axvline(x=modes1 - 0.5, color='red', linestyle='--', linewidth=0.8)
        ax_i.set_yscale('log')
        ax_i.grid(True, alpha=0.3)

        ax_j.plot(np.arange(final_profile.shape[1]), np.mean(final_profile, axis=0), 'o-', color='green')
        ax_j.set_title('j-Mode Profile (Final)')
        ax_j.set_xlabel(f'Mode j Index (0 to {modes2-1})')
        ax_j.set_ylabel('Avg Magnitude')
        ax_j.set_yscale('log')
        ax_j.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig_prof, save_path / f"{plot_file_prefix}_final_profiles.png")
        print(f"Finished visualizations for layer {layer_prefix}.")


def visualize_model_predictions(model, test_loader, n_samples, save_path):
    """ 
    Visualize model predictions vs ground truth. 
    """
    model.eval()
    try:
        # Attempt to get samples directly if dataset is indexable
        test_samples = [test_loader.dataset[i] for i in range(len(test_loader.dataset))]
    except TypeError: 
        # Fallback for IterableDataset or other non-indexable datasets
        print("Warning: test_loader.dataset is not directly indexable. Taking samples by iterating through loader.")
        test_samples = []
        for i_batch, batch in enumerate(test_loader):
            for j_sample in range(batch['x'].size(0)):
                if len(test_samples) < n_samples:
                    sample = {'x': batch['x'][j_sample], 'y': batch['y'][j_sample]}
                    test_samples.append(sample)
                else:
                    break 
            if len(test_samples) >= n_samples:
                break
        if not test_samples:
            print("Could not retrieve samples for prediction visualization.")
            return

    n_samples = min(n_samples, len(test_samples))
    if n_samples == 0:
        print("No samples to visualize for predictions.")
        return

    save_path = Path(save_path)

    fig, axs = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples), squeeze=False)
    fig.suptitle('Input, Ground Truth, and Prediction', fontsize=16)

    col_titles = ['Input (Permeability)', 'Ground Truth (Solution)', 'Prediction']
    for j_col, title in enumerate(col_titles):
        axs[0, j_col].set_title(title)

    with torch.no_grad():
        for i_sample_plot, data_sample in enumerate(test_samples[:n_samples]):
            
            # MODIFICATION: Removed data_processor.preprocess() call.
            # We now use the raw data directly from the sample.
            x_proc_raw = data_sample['x'].to(DEVICE) # Expected [C,H,W]
            y_proc_raw = data_sample['y']           # Expected [C,H,W], on CPU

            # Adapt input for the model (adds coordinate channels and batch dimension)
            x_adapted = adapt_input_data(x_proc_raw.unsqueeze(0), DEVICE)
            
            # Get model prediction
            out = model(x_adapted).squeeze(0).cpu()
            if out.dim() == 3 and out.shape[-1] == 1:
                out = out.squeeze(-1)

            # Prepare for plotting
            x_plot = x_proc_raw.squeeze().cpu().numpy()
            y_plot = y_proc_raw.squeeze().cpu().numpy()
            out_plot = out.numpy()

            # Shared normalization for GT and Prediction for consistent coloring
            vmin = min(y_plot.min(), out_plot.min())
            vmax = max(y_plot.max(), out_plot.max())
            norm = Normalize(vmin=vmin, vmax=vmax)

            # Plotting
            ims = [
                axs[i_sample_plot, 0].imshow(x_plot, cmap='gray'),
                axs[i_sample_plot, 1].imshow(y_plot, norm=norm, cmap='viridis'),
                axs[i_sample_plot, 2].imshow(out_plot, norm=norm, cmap='viridis')
            ]
            # Add colorbars and remove ticks
            for j_im, im_handle in enumerate(ims):
                fig.colorbar(im_handle, ax=axs[i_sample_plot, j_im], fraction=0.046, pad=0.04)
                axs[i_sample_plot, j_im].set_xticks([])
                axs[i_sample_plot, j_im].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, save_path / "predictions.png")
    print(f"Model prediction visualizations saved to {save_path / 'predictions.png'}")


# =====================================================================
# Main Execution Block
# =====================================================================

def main():
    """Main function to run the FNO training and visualization."""
    print("--- Configuration ---")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Modes: {MODES}, Width: {WIDTH}, Save Freq: {SAVE_FREQ}")
    print(f"Device: {DEVICE}, Output Dir: {OUTPUT_DIR.resolve()}")
    print("-" * 21)

    # --- Load Data ---
    data_path = './data'
    print(f"Loading Darcy Flow data with training resolution: {TRAIN_RESOLUTION}x{TRAIN_RESOLUTION}")
    try:
        # Create dataset
        dataset = DarcyDataset(root_dir=data_path,
                                n_train=N_TRAIN,
                                n_tests=[N_TEST],
                                batch_size=BATCH_SIZE,
                                test_batch_sizes=[BATCH_SIZE],
                                train_resolution=TRAIN_RESOLUTION,
                                test_resolutions=[TEST_RESOLUTION],
                                download=True)
        # Create data loaders
        train_loader = DataLoader(dataset.train_db,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)
        
        test_loaders = {}
        for res, test_bsize in zip([TEST_RESOLUTION], [BATCH_SIZE]):
            test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                           batch_size=test_bsize,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True)
        test_loader = test_loaders[TEST_RESOLUTION]

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the neuraloperator package is correctly installed and data is accessible.")
        return

    # --- Create Model ---
    print("Creating custom FNO2d model...")
    model = FNO2d(modes1=MODES, modes2=MODES, width=WIDTH).to(DEVICE)
    
    print('\n### MODEL STRUCTURE ###')
    print("Registered SpectralConv2d layers and their weights:")
    for name, module in model.named_modules():
        if isinstance(module, SpectralConv2d):
            print(f"  Layer: {name}")
            if hasattr(module, 'weights1'): print(f"    {name}.weights1 : {module.weights1.shape}")
            if hasattr(module, 'weights2'): print(f"    {name}.weights2 : {module.weights2.shape}")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model created with {n_params:,} trainable parameters.')


    # --- Train Model ---
    weight_evolution, trained_model, save_freq_actual = train_model(
        model, train_loader,
        epochs=EPOCHS, save_freq=SAVE_FREQ,
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        output_dir=OUTPUT_DIR
    )

    # --- Visualize Weights ---
    if weight_evolution: 
        print("\nVisualizing combined weight evolution for all spectral layers...")
        visualize_combined_weight_evolution(
            weight_evolution,
            save_path_base=OUTPUT_DIR / "weights", 
            save_freq=save_freq_actual
        )
    else:
        print("\nSkipping weight visualization as no spectral weights were tracked.")


    # --- Visualize Predictions ---
    print("\nVisualizing model predictions...")
    visualize_model_predictions(
        trained_model, test_loader,
        n_samples=N_VIS_SAMPLES,
        save_path=OUTPUT_DIR / "predictions" 
    )

    print("\nAnalysis complete!")
    print(f"- Weight evolution plots: '{(OUTPUT_DIR / 'weights').resolve()}'")
    print(f"- Prediction & metric plots: '{(OUTPUT_DIR / 'predictions').resolve()}'")

if __name__ == "__main__":
    main()
