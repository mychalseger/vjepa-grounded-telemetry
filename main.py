import os
import dill
import pickle
from itertools import islice
import torch


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

import h5py
#import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_pipeline import extract_and_cache_dataset
from baseline_model import RobotTrajectoryDataset, BaselineForecaster, train_baseline
from optimized_model import ContrastiveStudentMLP, ContrastiveForecaster, train_contrastive_model

# MuJoCo is imported dynamically in the simulation loop to check for library availability
# and support headless executions.

def generate_mujoco_xml(qpos_dim: int) -> str:
    """
    Generates a MuJoCo XML string tailored to the dimensionality of the robot state.
    - If dim is 2, creates a 2D planar pusher (x-y sliding body).
    - If dim is greater than 2, creates an N-link articulated robot arm with hinge joints.
    """
    if qpos_dim == 2:
        return """<mujoco model="planar_pusher">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.01"/>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
      <body name="pusher" pos="0 0 0.05">
        <joint name="slide_x" type="slide" axis="1 0 0"/>
        <joint name="slide_y" type="slide" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.05" rgba="0.8 0.2 0.2 1"/>
      </body>
    </worldbody>
</mujoco>"""
    else:
        xml = f"""<mujoco model="robot_arm">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.01"/>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>
      <body name="base" pos="0 0 0.1">
        <geom type="cylinder" size="0.2 0.05" rgba="0.1 0.1 0.1 1"/>
"""
        indent = "      "
        for i in range(qpos_dim):
            xml += f"""{indent}<body name="link_{i}" pos="0 0 0.2">
{indent}  <joint name="joint_{i}" type="hinge" axis="0 1 0" pos="0 0 0" range="-180 180"/>
{indent}  <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="{0.2 + 0.05*i:.2f} 0.5 0.8 1"/>
"""
            indent += "  "
            
        for i in range(qpos_dim):
            indent = indent[:-2]
            xml += f"{indent}</body>\n"
            
        xml += """    </body>
  </worldbody>
</mujoco>"""
    return xml

def run_evaluation_episode(
    baseline_model: torch.nn.Module,
    contrastive_model: torch.nn.Module,
    cache_path: str,
    history_len: int,
    device: str,
    eval_episode_id: int,
    blend_ratio: float = 0.5
):
    """
    Performs a closed-loop trajectory rollout in the MuJoCo simulator for a specific episode.
    Feeds predicted states back into the model history window to simulate
    compounding prediction errors (drift) and verify contrastive alignment stability.
    """
    import mujoco
    
    # 1. Load data statistics and test episode indices
    with h5py.File(cache_path, "r") as f:
        qpos_all = f["qpos"][:]
        qvel_all = f["qvel"][:]
        norm_qpos_all = f["norm_qpos"][:]
        norm_qvel_all = f["norm_qvel"][:]
        norm_action_all = f["norm_action"][:]
        episode_ids = f["episode_ids"][:]
        
        qpos_mean = f.attrs["qpos_mean"]
        qpos_std = f.attrs["qpos_std"]
        qvel_mean = f.attrs["qvel_mean"]
        qvel_std = f.attrs["qvel_std"]

    indices = np.where(episode_ids == eval_episode_id)[0]
    
    if len(indices) <= history_len:
        raise ValueError(f"Episode {eval_episode_id} does not have enough steps ({len(indices)}) for evaluation.")
                
    start_idx = indices[history_len - 1]
    rollout_len = min(50, len(indices) - history_len)
    
    true_traj = qpos_all[start_idx : start_idx + rollout_len]
    qpos_dim = qpos_all.shape[1]
    
    # Initialize MuJoCo physics model
    xml_str = generate_mujoco_xml(qpos_dim)
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    mj_data = mujoco.MjData(mj_model)
    
    # --- 1. Baseline Closed-Loop Rollout ---
    print(f"Running Baseline Closed-Loop Rollout in MuJoCo for Episode {eval_episode_id}...")
    baseline_traj = []
    # Seed initial history window from dataset
    hist_qpos = norm_qpos_all[start_idx - history_len + 1 : start_idx + 1].copy()
    hist_qvel = norm_qvel_all[start_idx - history_len + 1 : start_idx + 1].copy()
    
    baseline_model.eval()
    with torch.no_grad():
        for t in range(rollout_len):
            joint_input = np.concatenate([hist_qpos, hist_qvel], axis=-1)
            joint_input_t = torch.tensor(joint_input, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict future qpos sequence, take the immediate next step (index 0)
            pred = baseline_model(joint_input_t).cpu().numpy()[0, 0]
            baseline_traj.append(pred)
            
            # Set joint states in simulator and step physics
            mj_data.qpos[:] = pred
            mujoco.mj_step(mj_model, mj_data)
            
            # Fetch feedback sensor values from MuJoCo
            feedback_qpos = mj_data.qpos[:].copy()
            feedback_qvel = mj_data.qvel[:qpos_dim].copy()
            
            # Normalize simulator outputs to update history window
            norm_fb_qpos = (feedback_qpos - qpos_mean) / qpos_std
            norm_fb_qvel = (feedback_qvel - qvel_mean) / qvel_std
            
            # Shift and append to history window
            hist_qpos = np.roll(hist_qpos, -1, axis=0)
            hist_qpos[-1] = norm_fb_qpos
            hist_qvel = np.roll(hist_qvel, -1, axis=0)
            hist_qvel[-1] = norm_fb_qvel
 
    # --- 2. Contrastive Aligned Closed-Loop Rollout with Visual Anchor Guidance ---
    print(f"Running Contrastive Aligned Closed-Loop Rollout in MuJoCo for Episode {eval_episode_id}...")
    contrastive_traj = []
    hist_qpos = norm_qpos_all[start_idx - history_len + 1 : start_idx + 1].copy()
    hist_qvel = norm_qvel_all[start_idx - history_len + 1 : start_idx + 1].copy()
    
    contrastive_model.eval()
    with torch.no_grad():
        for t in range(rollout_len):
            joint_input = np.concatenate([hist_qpos, hist_qvel], axis=-1)
            joint_input_t = torch.tensor(joint_input, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Fetch the current step's action command (normalized)
            action_t = torch.tensor(norm_action_all[start_idx + t], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Pass joint inputs, next action, visual anchor (None), and blend ratio (1.0)
            pred = contrastive_model(joint_input_t, action=action_t, visual_anchor=None, blend_weight=1.0).cpu().numpy()[0, 0]
            contrastive_traj.append(pred)
            
            # Set joint states in simulator and step physics
            mj_data.qpos[:] = pred
            mujoco.mj_step(mj_model, mj_data)
            
            feedback_qpos = mj_data.qpos[:].copy()
            feedback_qvel = mj_data.qvel[:qpos_dim].copy()
            
            norm_fb_qpos = (feedback_qpos - qpos_mean) / qpos_std
            norm_fb_qvel = (feedback_qvel - qvel_mean) / qvel_std
            
            # Shift and append to history window
            hist_qpos = np.roll(hist_qpos, -1, axis=0)
            hist_qpos[-1] = norm_fb_qpos
            hist_qvel = np.roll(hist_qvel, -1, axis=0)
            hist_qvel[-1] = norm_fb_qvel
            
    return np.array(true_traj), np.array(baseline_traj), np.array(contrastive_traj)

def save_trajectory_animation(true_traj, baseline_traj, contrastive_traj, ep_id):
    """
    Generates a beautiful 2D path tracking GIF animation comparing 
    Ground Truth, Baseline, and Contrastive Aligned trajectories.
    """
    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(7, 7))
    
    all_points = np.concatenate([true_traj, baseline_traj, contrastive_traj], axis=0)
    x_min, y_min = all_points.min(axis=0) - 0.05
    x_max, y_max = all_points.max(axis=0) + 0.05
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Planar Pusher Rollout Path - Episode {ep_id}")
    ax.set_xlabel("Pusher Position X")
    ax.set_ylabel("Pusher Position Y")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Pre-draw static lines in faint color
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'k-', alpha=0.15)
    ax.plot(baseline_traj[:, 0], baseline_traj[:, 1], 'r--', alpha=0.15)
    ax.plot(contrastive_traj[:, 0], contrastive_traj[:, 1], 'g-', alpha=0.15)
    
    # Animated objects
    true_line, = ax.plot([], [], 'k-', linewidth=2, label="Ground Truth")
    base_line, = ax.plot([], [], 'r--', linewidth=1.5, label="Baseline LSTM")
    contr_line, = ax.plot([], [], 'g-', linewidth=2, label="Contrastive Aligned")
    
    true_dot, = ax.plot([], [], 'ko', markersize=8)
    base_dot, = ax.plot([], [], 'ro', markersize=8)
    contr_dot, = ax.plot([], [], 'go', markersize=8)
    
    ax.legend(loc="upper right")
    
    def update(frame):
        true_line.set_data(true_traj[:frame, 0], true_traj[:frame, 1])
        base_line.set_data(baseline_traj[:frame, 0], baseline_traj[:frame, 1])
        contr_line.set_data(contrastive_traj[:frame, 0], contrastive_traj[:frame, 1])
        
        true_dot.set_data([true_traj[frame-1, 0]], [true_traj[frame-1, 1]])
        base_dot.set_data([baseline_traj[frame-1, 0]], [baseline_traj[frame-1, 1]])
        contr_dot.set_data([contrastive_traj[frame-1, 0]], [contrastive_traj[frame-1, 1]])
        return true_line, base_line, contr_line, true_dot, base_dot, contr_dot
        
    num_frames = len(true_traj)
    ani = animation.FuncAnimation(fig, update, frames=range(1, num_frames + 1), blit=True)
    
    gif_path = f"trajectory_animation_ep_{ep_id}.gif"
    ani.save(gif_path, writer='pillow', fps=10)
    plt.close()
    print(f"Successfully saved trajectory animation to {gif_path}")
    return gif_path

def run_experiment_pipeline(lambda_weight: float, blend_ratio: float, schedule_epoch: int, lr: float, epochs: int = 500, jitter_enabled: bool = False) -> dict:
    """
    Trains the baseline and contrastive forecasters, evaluates their rollout
    accuracy in MuJoCo, and returns a dictionary of the drift results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Running Experiment: lambda={lambda_weight}, blend={blend_ratio}, schedule_epoch={schedule_epoch}, lr={lr}, epochs={epochs}, jitter={jitter_enabled} ---")
    
    dataset_name = "lerobot/pusht"
    cache_path = "feature_cache.h5"
    history_len = 40
    pred_len = 5
    batch_size = 1024
    
    # Step 1: Preprocessing & Cache Generation
    if not os.path.exists(cache_path):
        print("Feature cache not found. Running offline image feature extraction...")
        extract_and_cache_dataset(
            dataset_name=dataset_name,
            cache_path=cache_path,
            batch_size=batch_size,
            device=device,
            limit_frames=None
        )
    else:
        print(f"Found existing feature cache at {cache_path}. Skipping extraction.")
        
    # Get unique episodes to split train, validation, and test sets
    with h5py.File(cache_path, "r") as f:
        episode_ids = f["episode_ids"][:]
    unique_eps = np.unique(episode_ids)
    num_eps = len(unique_eps)
    
    # 80% train, 10% validation, 10% test
    if num_eps >= 3:
        test_split_idx = max(2, int(num_eps * 0.9))
        val_split_idx = max(1, int(num_eps * 0.8))
        train_episodes = unique_eps[:val_split_idx]
        val_episodes = unique_eps[val_split_idx:test_split_idx]
        test_episodes = unique_eps[test_split_idx:]
    else:
        train_episodes = unique_eps
        val_episodes = unique_eps
        test_episodes = unique_eps
        
    print(f"Split Summary: Total Episodes = {num_eps} | Training = {len(train_episodes)} | Validation = {len(val_episodes)} | Testing = {len(test_episodes)}")
    print(f"Train Episode IDs: {train_episodes}")
    print(f"Val Episode IDs: {val_episodes}")
    print(f"Test Episode IDs: {test_episodes}")
        
    # Step 2: Load Cached Trajectory Dataset (Train split only)
    dataset = RobotTrajectoryDataset(
        cache_path=cache_path,
        history_len=history_len,
        pred_len=pred_len,
        allowed_episodes=train_episodes
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Load Validation Dataset (Val split only)
    val_dataset = RobotTrajectoryDataset(
        cache_path=cache_path,
        history_len=history_len,
        pred_len=pred_len,
        allowed_episodes=val_episodes
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_episodes) > 0 else None
    
    # Identify joint sizes
    with h5py.File(cache_path, "r") as f:
        qpos_dim = f["qpos"].shape[1]
        qvel_dim = f["qvel"].shape[1]
        
    input_dim = history_len * (qpos_dim + qvel_dim)
    output_dim = qpos_dim
    
    # Step 3: Train Baseline Model
    baseline_forecaster = BaselineForecaster(
        input_dim=qpos_dim + qvel_dim,
        output_dim=output_dim,
        hidden_dim=256,
        num_layers=2,
        pred_len=pred_len
    )
    baseline_forecaster = train_baseline(
        model=baseline_forecaster,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        device=device,
        jitter_enabled=jitter_enabled,
        val_dataloader=val_dataloader,
        patience=15
    )
    
    # Step 4: Train Contrastive Aligned Model (with torch.compile)
    student_mlp = ContrastiveStudentMLP(
        input_dim=input_dim,
        latent_dim=1024,
        hidden_dim=512,
        action_dim=qpos_dim
    )
    contrastive_forecaster = ContrastiveForecaster(
        student=student_mlp,
        output_dim=output_dim,
        pred_len=pred_len
    )
    contrastive_forecaster = train_contrastive_model(
        model=contrastive_forecaster,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        temperature=0.07,
        lambda_weight=lambda_weight,
        target_blend_ratio=blend_ratio,
        schedule_epoch=schedule_epoch,
        device=device,
        jitter_enabled=jitter_enabled,
        val_dataloader=val_dataloader,
        patience=25
    )
    
    # Step 5: MuJoCo Physics Evaluation & Drift Calculation for each test episode
    test_results = {}
    import json
    
    for ep_id in test_episodes:
        try:
            true_traj, baseline_traj, contrastive_traj = run_evaluation_episode(
                baseline_model=baseline_forecaster,
                contrastive_model=contrastive_forecaster,
                cache_path=cache_path,
                history_len=history_len,
                device=device,
                eval_episode_id=ep_id,
                blend_ratio=blend_ratio
            )
            
            # Save a comparison plot for EACH test episode
            plt.figure(figsize=(12, 6))
            plt.plot(true_traj[:, 0], 'k-', label="Ground Truth (Dataset)", linewidth=2)
            plt.plot(baseline_traj[:, 0], 'r--', label="Baseline LSTM Forecaster (Blind)", linewidth=1.5)
            plt.plot(contrastive_traj[:, 0], 'g-', label="Contrastive Aligned Forecaster (V-JEPA Grounded)", linewidth=2.0)
            plt.title(f"Planar Pusher Rollout - Episode {ep_id} (lambda={lambda_weight}, blend={blend_ratio})")
            plt.xlabel("Time Step (0.01s)")
            plt.ylabel("Pusher Position")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            plot_filename = f"trajectory_comparison_ep_{ep_id}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()
            
            # Save trajectory animation GIF
            gif_filename = save_trajectory_animation(true_traj, baseline_traj, contrastive_traj, ep_id)
            
            # Compute final quantitative drift metrics
            b_drift = float(np.mean(np.linalg.norm(true_traj - baseline_traj, axis=-1)))
            c_drift = float(np.mean(np.linalg.norm(true_traj - contrastive_traj, axis=-1)))
            improvement = (b_drift - c_drift) / b_drift * 100 if b_drift > 0 else 0
            
            test_results[str(ep_id)] = {
                "baseline_drift": b_drift,
                "contrastive_drift": c_drift,
                "improvement_percent": improvement,
                "plot_path": plot_filename,
                "gif_path": gif_filename
            }
            
            print(f"Episode {ep_id} | Baseline: {b_drift:.6f} | Contrastive: {c_drift:.6f} | Improvement: {improvement:.2f}%")
            
        except Exception as e:
            print(f"MuJoCo evaluation / plotting failed for episode {ep_id}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save the optimization/eval results to a json file
    with open("episode_evaluation_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
        
    print("\n" + "="*50)
    print("EXPERIMENT EVALUATION RESULTS (ON HELD-OUT TEST EPISODES):")
    for ep_id, metrics in test_results.items():
        print(f"Episode {ep_id}:")
        print(f"  Baseline LSTM Trajectory Drift (Average L2 Error): {metrics['baseline_drift']:.6f}")
        print(f"  Contrastive Aligned Trajectory Drift (Average L2 Error): {metrics['contrastive_drift']:.6f}")
        print(f"  Visual Grounding Drift Reduction: {metrics['improvement_percent']:.2f}%")
    print("="*50 + "\n")
        
    avg_baseline_drift = np.mean([m["baseline_drift"] for m in test_results.values()]) if test_results else 999.0
    avg_contrastive_drift = np.mean([m["contrastive_drift"] for m in test_results.values()]) if test_results else 999.0
    
    return {
        "baseline_drift": avg_baseline_drift,
        "contrastive_drift": avg_contrastive_drift,
        "test_results": test_results
    }

def main():
    # Run pipeline with default parameters
    results = run_experiment_pipeline(
        lambda_weight=0.25,
        blend_ratio=0.4,
        schedule_epoch=30,
        lr=1e-3,
        epochs=120,
        jitter_enabled=True
    )
    print(f"Run completed. Results: {results}")

if __name__ == "__main__":
    main()
