import os
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModel
from tqdm import tqdm
import numpy as np
import dill
import pickle
from itertools import islice

# Python 3.14 compatibility patch for dill serialization in HF datasets
def patched_batch_setitems(self, items, obj=None):
    save = self.save
    write = self.write
    if not self.bin:
        for k, v in items:
            save(k)
            save(v)
            write(pickle.SETITEM)
        return
    batch_size = getattr(self, '_BATCHSIZE', 1000)
    it = iter(items)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        if len(batch) != 1:
            write(pickle.MARK)
            for k, v in batch:
                save(k)
                save(v)
            write(pickle.SETITEMS)
        else:
            k, v = batch[0]
            save(k)
            save(v)
            write(pickle.SETITEM)

dill._dill.Pickler._batch_setitems = patched_batch_setitems
if hasattr(dill, 'Pickler'):
    dill.Pickler._batch_setitems = patched_batch_setitems

# Patch datasets.utils._dill.Pickler._batch_setitems for Python 3.14
import datasets.utils._dill
orig_datasets_batch_setitems = datasets.utils._dill.Pickler._batch_setitems

def patched_datasets_batch_setitems(self, items, *args, **kwargs):
    return orig_datasets_batch_setitems(self, items)

datasets.utils._dill.Pickler._batch_setitems = patched_datasets_batch_setitems

# Ensure HuggingFace Hub doesn't complain about locks
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Patch lerobot video decoding for Windows (pure-pyav fallback with memory cache)
import av
from lerobot.datasets import video_utils
import lerobot.datasets.lerobot_dataset

_DECODED_VIDEO_CACHE = {}

def decode_video_frames_pure_pyav(video_path, timestamps, tolerance_s=0.1, backend=None):
    video_path_str = str(video_path)
    if video_path_str in _DECODED_VIDEO_CACHE:
        loaded_frames, loaded_ts = _DECODED_VIDEO_CACHE[video_path_str]
    else:
        container = av.open(video_path_str)
        frames = []
        frame_timestamps = []
        
        for frame in container.decode(video=0):
            ts = frame.time
            img_np = frame.to_ndarray(format="rgb24")
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # [3, H, W]
            frames.append(img_tensor)
            frame_timestamps.append(ts)
            
        container.close()
        
        loaded_frames = torch.stack(frames)
        loaded_ts = torch.tensor(frame_timestamps, dtype=torch.float32)
        _DECODED_VIDEO_CACHE[video_path_str] = (loaded_frames, loaded_ts)
        
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    
    # Calculate distances
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)
    
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = closest_frames.type(torch.float32) / 255.0
    return closest_frames

video_utils.decode_video_frames = decode_video_frames_pure_pyav
lerobot.datasets.lerobot_dataset.decode_video_frames = decode_video_frames_pure_pyav

class FP8Linear(nn.Module):
    """
    A custom FP8 Linear Layer designed for NVIDIA RTX 5080 (Blackwell architecture).
    It quantizes weight tensors to torch.float8_e4m3fn precision and computes 
    matrix multiplications using PyTorch's native FP8 support on CUDA.
    """
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        # Store in FP8
        self.weight = nn.Parameter(base_linear.weight.data.to(torch.float8_e4m3fn), requires_grad=False)
        self.bias = nn.Parameter(base_linear.bias.data) if base_linear.bias is not None else None
        
        # Scaling factors for quantization (FP8 uses scaling to maximize dynamic range)
        # E4M3 range is [-240, 240]
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("input_scale", torch.tensor(1.0))
        
        # Dynamically calculate weight scale based on max absolute value
        max_val = base_linear.weight.data.abs().max().clamp(min=1e-5)
        self.weight_scale.copy_(max_val / 240.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original shape to handle 3D tensors (torch._scaled_mm expects 2D matrices)
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(-1, x.size(-1))
        else:
            x_flat = x

        # Dynamically calculate input scale with epsilon to prevent division by zero/precision underflow
        max_val = x_flat.abs().max().clamp(min=1e-5)
        input_scale = (max_val + 1e-5) / 240.0

        # Scale and quantize input to FP8
        x_scaled = (x_flat / input_scale).clamp(-240.0, 240.0)
        x_f8 = x_scaled.to(torch.float8_e4m3fn)
        
        if x.is_cuda:
            # Scaled matrix multiplication utilizing Blackwell FP8 Tensor Cores
            # torch._scaled_mm returns output in high precision matching x
            out = torch._scaled_mm(
                x_f8,
                self.weight.t(),
                scale_a=input_scale,
                scale_b=self.weight_scale,
                out_dtype=x.dtype
            )
        else:
            # Fallback for CPU testing
            out = F.linear(
                x_f8.to(x.dtype) * input_scale, 
                self.weight.to(x.dtype) * self.weight_scale
            )
            
        if self.bias is not None:
            out = out + self.bias

        # Reshape back to original dimensions if we flattened a 3D/nD tensor
        if x.dim() > 2:
            out = out.reshape(*orig_shape[:-1], -1)
            
        return out

def replace_linear_with_fp8(model: nn.Module):
    """
    Recursively replaces nn.Linear layers with FP8Linear,
    skipping nn.MultiheadAttention to prevent attention kernel dtype conflicts.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.MultiheadAttention):
            continue
        elif isinstance(child, nn.Linear):
            setattr(model, name, FP8Linear(child))
        else:
            replace_linear_with_fp8(child)

class FP8VJEPAEncoder(nn.Module):
    """
    V-JEPA wrapper that runs visual feature extraction in FP8 precision.
    Outputs a 1024-dimensional pooled visual latent representation.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained V-JEPA Large model (default embedding dimension is 1024)
        self.model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        self.model.eval()
        
        # Replace all linear layers with custom FP8 implementations
        replace_linear_with_fp8(self.model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W], where T=64, C=3, H=256, W=256
        # V-JEPA expects pixel_values_videos: [B, T, C, H, W]
        outputs = self.model(pixel_values_videos=x)
        # last_hidden_state shape: [B, 8192, 1024]
        # Mean pool over the token dimension to get [B, 1024]
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled

class LeRobotSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)
    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError()
        return getattr(self.dataset, name)

def extract_and_cache_dataset(
    dataset_name: str = "lerobot/pusht",
    cache_path: str = "feature_cache.h5",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    limit_frames: int = None
):
    """
    Streams a LeRobot dataset, runs FP8 visual feature extraction, and caches
    features along with robot qpos & qvel to disk.
    """
    print(f"Loading LeRobot dataset: {dataset_name}...")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets import LeRobotDataset
    
    # Load dataset
    dataset = LeRobotDataset(dataset_name, video_backend="pyav")
    num_frames = len(dataset)
    print(f"Dataset loaded. Total frames: {num_frames}, Features: {list(dataset.features.keys())}")
    
    if limit_frames is not None:
        dataset = LeRobotSubset(dataset, range(min(limit_frames, len(dataset))))
        print(f"Slicing dataset to first {len(dataset)} frames.")
    
    # Identify keys
    image_key = None
    state_key = "observation.state"
    
    # Find any image keys
    for k in dataset.features.keys():
        if "image" in k:
            image_key = k
            break
            
    if image_key is None:
        raise ValueError("No image key found in LeRobot dataset features!")
        
    print(f"Using image key: {image_key} and state key: {state_key}")
    
    # Setup data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # Setup FP8 Encoder
    print("Initializing Visual Foundation Encoder in FP8 precision...")
    encoder = FP8VJEPAEncoder().to(device)
    encoder.eval()
    
    # Transform for images (V-JEPA expects [3, 256, 256] normalized to ImageNet stats)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Variables to hold extracted features and telemetry
    all_images = []
    all_qpos = []
    all_qvel = []
    all_action = []
    all_episodes = []
    
    print("Extracting physical visual frames and joint telemetry...")
    with torch.no_grad():
        # Keep track of previous state for velocity estimation if qvel is not directly stored
        prev_qpos = None
        
        for batch in tqdm(dataloader):
            # 1. Process Images (Keep raw CPU tensors in memory to avoid OOM)
            images = batch[image_key].cpu()
            all_images.append(images)
            
            # 2. Process Telemetry (qpos, qvel) and actions
            qpos = batch[state_key].float()
            action = batch["action"].float()
            
            # Compute qvel as finite differences if not explicitly present
            # LeRobot datasets typically have joint velocities or we can compute them
            qvel = torch.zeros_like(qpos)
            if "observation.velocity" in batch:
                qvel = batch["observation.velocity"].float()
            else:
                # Finite difference approximation
                # qvel_t = qpos_t - qpos_{t-1}
                qvel_np = np.zeros_like(qpos.numpy())
                qpos_np = qpos.numpy()
                for i in range(len(qpos_np)):
                    if prev_qpos is not None:
                        qvel_np[i] = qpos_np[i] - prev_qpos
                    prev_qpos = qpos_np[i]
                qvel = torch.tensor(qvel_np, dtype=torch.float32)
                
            all_qpos.append(qpos.cpu().numpy())
            all_qvel.append(qvel.cpu().numpy())
            all_action.append(action.cpu().numpy())
            
            # 3. Track episode metadata to avoid boundary bleed during windowing
            episode_index = batch["episode_index"].numpy()
            all_episodes.append(episode_index)
            
    # Concatenate results
    all_images = torch.cat(all_images, dim=0) # [N, 3, H, W] on CPU
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_qvel = np.concatenate(all_qvel, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    all_episodes = np.concatenate(all_episodes, axis=0)
    
    N = len(all_episodes)
    
    # Extract features using V-JEPA in batches with on-the-fly clip generation
    print("Running V-JEPA visual encoder over clips (on-the-fly sliding windows to save memory)...")
    all_latents = []
    vjepa_batch_size = 8  # Safe default to avoid VRAM swapping on RTX 5080
    for idx in tqdm(range(0, N, vjepa_batch_size)):
        # Construct the batch of raw clips on-the-fly
        batch_clips = []
        for i in range(idx, min(idx + vjepa_batch_size, N)):
            ep_id = all_episodes[i]
            # Find start index of current episode
            start_idx = i
            while start_idx > 0 and all_episodes[start_idx - 1] == ep_id:
                start_idx -= 1
            
            # Generate indices for 64-frame clip ending at i (pad with first frame if history < 64)
            clip_indices = [max(start_idx, step) for step in range(i - 63, i + 1)]
            clip_raw = all_images[clip_indices] # Shape: [64, 3, H, W] on CPU
            batch_clips.append(clip_raw)
            
        # Send raw clips directly to GPU (low PCIe bandwidth)
        batch_tensor = torch.stack(batch_clips).to(device) # Shape: [B, 64, 3, H, W] on GPU
        
        # GPU-accelerated float conversion, resize, and normalization
        B, T, C, H, W = batch_tensor.shape
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        if batch_tensor.dtype == torch.uint8:
            batch_tensor = batch_tensor.float() / 255.0
        else:
            batch_tensor = batch_tensor.float()
            
        # Preprocess on GPU
        batch_tensor = preprocess(batch_tensor)
        
        # Reshape back to video format [B, T, C, 256, 256]
        batch_tensor = batch_tensor.view(B, T, C, 256, 256)
        
        with torch.no_grad():
            latents = encoder(batch_tensor) # Shape: [B, 1024]
            all_latents.append(latents.cpu().numpy())
            
    all_latents = np.concatenate(all_latents, axis=0)
    
    # Calculate normalization stats using only the train split
    unique_eps = np.unique(all_episodes)
    num_eps = len(unique_eps)
    if num_eps >= 3:
        val_split_idx = max(1, int(num_eps * 0.8))
        train_episodes = unique_eps[:val_split_idx]
    else:
        train_episodes = unique_eps
        
    train_mask = np.isin(all_episodes, train_episodes)
    
    qpos_mean = np.mean(all_qpos[train_mask], axis=0)
    qpos_std = np.std(all_qpos[train_mask], axis=0) + 1e-6
    qvel_mean = np.mean(all_qvel[train_mask], axis=0)
    qvel_std = np.std(all_qvel[train_mask], axis=0) + 1e-6
    action_mean = np.mean(all_action[train_mask], axis=0)
    action_std = np.std(all_action[train_mask], axis=0) + 1e-6
    
    # Normalize telemetry and actions
    norm_qpos = (all_qpos - qpos_mean) / qpos_std
    norm_qvel = (all_qvel - qvel_mean) / qvel_std
    norm_action = (all_action - action_mean) / action_std
    
    print(f"Caching features to disk: {cache_path}...")
    with h5py.File(cache_path, "w") as f:
        f.create_dataset("visual_latents", data=all_latents)
        f.create_dataset("qpos", data=all_qpos)
        f.create_dataset("qvel", data=all_qvel)
        f.create_dataset("action", data=all_action)
        f.create_dataset("norm_qpos", data=norm_qpos)
        f.create_dataset("norm_qvel", data=norm_qvel)
        f.create_dataset("norm_action", data=norm_action)
        f.create_dataset("episode_ids", data=all_episodes)
        
        # Save normalization stats
        f.attrs["qpos_mean"] = qpos_mean
        f.attrs["qpos_std"] = qpos_std
        f.attrs["qvel_mean"] = qvel_mean
        f.attrs["qvel_std"] = qvel_std
        f.attrs["action_mean"] = action_mean
        f.attrs["action_std"] = action_std
        
    print("Preprocessing completed and cached successfully.")

if __name__ == "__main__":
    # Test execution
    extract_and_cache_dataset()
