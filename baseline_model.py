import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RobotTrajectoryDataset(Dataset):
    """
    Loads joint trajectory history and forecasts future trajectories from the cached HDF5 database.
    Prevents sequence boundary bleed by ensuring all windows lie entirely within a single episode.
    """
    def __init__(self, cache_path: str = "feature_cache.h5", history_len: int = 40, pred_len: int = 5, allowed_episodes=None):
        self.cache_path = cache_path
        self.history_len = history_len
        self.pred_len = pred_len
        self.allowed_episodes = set(allowed_episodes) if allowed_episodes is not None else None
        
        # Load indices and metadata
        with h5py.File(cache_path, "r") as f:
            self.episode_ids = f["episode_ids"][:]
            self.dataset_size = len(self.episode_ids)
            
        # Precompute valid starting indices where the entire history and prediction window
        # belong to the same episode.
        self.valid_indices = []
        for idx in range(self.history_len - 1, self.dataset_size - self.pred_len):
            start_ep = self.episode_ids[idx - self.history_len + 1]
            end_ep = self.episode_ids[idx + self.pred_len]
            if start_ep == end_ep:
                if self.allowed_episodes is None or start_ep in self.allowed_episodes:
                    self.valid_indices.append(idx)
                
        print(f"Dataset compiled. Total frames: {self.dataset_size}, Valid sequential windows: {len(self.valid_indices)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        # Retrieve actual index in cached dataset
        target_idx = self.valid_indices[idx]
        
        with h5py.File(self.cache_path, "r") as f:
            # History inputs: qpos and qvel
            qpos_seq = f["norm_qpos"][target_idx - self.history_len + 1 : target_idx + 1]
            qvel_seq = f["norm_qvel"][target_idx - self.history_len + 1 : target_idx + 1]
            
            # Future target: next raw qpos coordinates (we predict actual coordinates for MuJoCo control)
            future_qpos_seq = f["qpos"][target_idx + 1 : target_idx + 1 + self.pred_len]
            
            # Anchor: visual world model physical latent corresponding to the CURRENT step (anchor step t)
            visual_latent = f["visual_latents"][target_idx]
            
            # Action command corresponding to the CURRENT step (t)
            action = f["norm_action"][target_idx]
            
        # Concatenate normalized qpos and qvel to create joint states
        # Input shape: [history_len, qpos_dim + qvel_dim]
        joint_state_seq = np.concatenate([qpos_seq, qvel_seq], axis=-1)
        
        return {
            "joint_inputs": torch.tensor(joint_state_seq, dtype=torch.float32),
            "future_qpos_targets": torch.tensor(future_qpos_seq, dtype=torch.float32),
            "visual_latent_anchor": torch.tensor(visual_latent, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32)
        }

class BaselineForecaster(nn.Module):
    """
    Standard Time-Series Forecasting Network (LSTM Seq2Seq).
    Takes a history of joint states and directly forecasts the future trajectory of qpos.
    Operates completely blind to visual physics context, trained solely on MSE.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 2, pred_len: int = 10):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Decoder LSTM that rolls out the prediction horizon
        self.decoder_cell = nn.LSTMCell(
            input_size=output_dim,
            hidden_size=hidden_dim
        )
        
        # Projection output layer mapping hidden state to joint qpos space
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, joint_history: torch.Tensor) -> torch.Tensor:
        # joint_history shape: [Batch, history_len, input_dim]
        # output shape: [Batch, pred_len, output_dim]
        
        batch_size = joint_history.size(0)
        
        # Run encoder
        _, (h_n, c_n) = self.encoder(joint_history)
        
        # We take the top layer state for decoding
        h_t = h_n[-1]
        c_t = c_n[-1]
        
        # Initialize decoder input with the last qpos in the history window
        # (qpos is the first segment of input_dim, which typically consists of [qpos, qvel])
        # We can extract the final step's qpos: shape [Batch, output_dim]
        decoder_input = joint_history[:, -1, :self.output_dim]
        
        predictions = []
        for _ in range(self.pred_len):
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            pred_step = self.out_proj(h_t)
            predictions.append(pred_step.unsqueeze(1))
            # Autoregressive feeding: the output prediction becomes the next input
            decoder_input = pred_step
            
        return torch.cat(predictions, dim=1)
 
def train_baseline(
    model: nn.Module, 
    dataloader: DataLoader, 
    epochs: int = 100, 
    lr: float = 1e-3, 
    device: str = "cuda", 
    jitter_enabled: bool = False,
    val_dataloader: DataLoader = None,
    patience: int = 15
):
    epochs = 200
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_state_dict = None
    epochs_no_improve = 0
    
    print("Starting training of Baseline LSTM Forecaster...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch["joint_inputs"].to(device)
            targets = batch["future_qpos_targets"].to(device)
            
            if jitter_enabled:
                inputs = inputs + torch.randn_like(inputs) * 0.02
                
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        
        # Validation and early stopping
        val_loss_str = ""
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch["joint_inputs"].to(device)
                    targets = batch["future_qpos_targets"].to(device)
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_dataloader.dataset)
            val_loss_str = f" | Val MSE Loss: {val_loss:.6f}"
            
            if val_loss < best_loss:
                best_loss = val_loss
                import copy
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Epoch {epoch+1}/{epochs} | Early stopping baseline training. Restoring best model with Val Loss: {best_loss:.6f}")
                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                break
        
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1 or (val_dataloader is not None and epochs_no_improve == 0):
            print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | Training MSE Loss: {epoch_loss:.6f}{val_loss_str}")
        
        scheduler.step()
        
    return model
        
    return model
