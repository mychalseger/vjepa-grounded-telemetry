import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ContrastiveStudentMLP(nn.Module):
    """
    Student Network mapping mechanical joint telemetry AND the target action
    into the physical visual latent space (Action-Conditioned World Model).
    """
    def __init__(self, input_dim: int, latent_dim: int = 1024, hidden_dim: int = 512, action_dim: int = 2, history_len: int = 40, pos_dim: int = 16):
        super().__init__()
        self.history_len = history_len
        self.pos_dim = pos_dim
        self.action_dim = action_dim
        
        # Precompute fixed sinusoidal positional embeddings representing the sequence time-step
        import math
        pe = torch.zeros(history_len, pos_dim)
        position = torch.arange(0, history_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe) # Shape: [history_len, pos_dim]
        
        # joint_dim is the feature size of a single step
        self.joint_dim = input_dim // history_len
        total_input_dim = history_len * (self.joint_dim + pos_dim) + action_dim
        
        # Input layer mapping flattened state + action to hidden space
        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Projection layer mapping to 1024-dim visual world model latent space
        self.projection = nn.Linear(hidden_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # x shape: [B, history_len * joint_dim] or [B, history_len, joint_dim]
        if x.dim() == 2:
            x = x.view(x.size(0), self.history_len, -1)
            
        batch_size = x.size(0)
        
        # Broadcast pe buffer to match batch dimension
        pe_broadcast = self.pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate scaled positional embeddings (0.1 weight) to prevent coordinate offset dominance
        x_pe = torch.cat([x, pe_broadcast * 0.1], dim=-1) # Shape: [B, history_len, joint_dim + pos_dim]
        x_flat = x_pe.reshape(batch_size, -1)
        
        combined = torch.cat([x_flat, action], dim=-1)
        h = self.input_layer(combined)
        
        # Residual connections
        h = h + self.block1(h)
        h = h + self.block2(h)
        
        out = self.projection(h)
        return self.layer_norm(out)

class ContrastiveDecoder(nn.Module):
    """
    Sequence-aware LSTM decoder head mapping the student's aligned 1024-dim 
    representation down to the future joint qpos trajectory.
    Includes a Multi-Scale Manifold Self-Attention layer over the hidden states.
    """
    def __init__(self, output_dim: int, hidden_dim: int = 256, pred_len: int = 5):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Projections to initialize LSTM hidden and cell state from 1024-dim aligned representation
        self.h_init_proj = nn.Linear(1024, hidden_dim)
        self.c_init_proj = nn.Linear(1024, hidden_dim)
        
        # Decoder LSTM Cell
        self.decoder_cell = nn.LSTMCell(
            input_size=output_dim,
            hidden_size=hidden_dim
        )
        
        # Query, Key, Value projections for the scaled dot-product attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, aligned_rep: torch.Tensor, last_qpos: torch.Tensor) -> torch.Tensor:
        import math
        # Initialize hidden and cell states from aligned representations
        h_t = self.h_init_proj(aligned_rep)
        c_t = self.c_init_proj(aligned_rep)
        
        decoder_input = last_qpos
        hidden_history = []
        predictions = []
        for _ in range(self.pred_len):
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            hidden_history.append(h_t.unsqueeze(1)) # Shape: [B, 1, H]
            
            # Stack all hidden states generated so far
            H_seq = torch.cat(hidden_history, dim=1) # Shape: [B, current_len, H]
            
            # Scaled Dot-Product Attention: Query is current h_t, Keys/Values are H_seq
            Q = self.q_proj(h_t.unsqueeze(1))
            K = self.k_proj(H_seq)
            V = self.v_proj(H_seq)
            
            # Attention scores: Q @ K^T / sqrt(H) -> [B, 1, current_len]
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Context shape: [B, H]
            attn_out = torch.matmul(attn_weights, V).squeeze(1)
            
            # Residual connection
            h_attn = attn_out + h_t
            
            pred_step = self.out_proj(h_attn)
            predictions.append(pred_step.unsqueeze(1))
            # Autoregressive feeding: predicted step becomes next input
            decoder_input = pred_step
            
        return torch.cat(predictions, dim=1)

class ContrastiveForecaster(nn.Module):
    """
    Forecasting wrapper model that maps the student's aligned 1024-dim representation 
    down to the future joint qpos trajectory.
    """
    def __init__(self, student: ContrastiveStudentMLP, output_dim: int, pred_len: int = 10):
        super().__init__()
        self.student = student
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # Deeper sequence-aware decoder to prevent flatlining
        self.decoder = ContrastiveDecoder(
            output_dim=output_dim,
            hidden_dim=256,
            pred_len=pred_len
        )
        
    def forward(self, joint_history: torch.Tensor, action: torch.Tensor, visual_anchor: torch.Tensor = None, blend_weight: float = 1.0) -> torch.Tensor:
        # 1. Project joint history + action to aligned latent space [B, 1024]
        aligned_rep = self.student(joint_history, action)
        
        # 2. Optionally blend with the visual anchor embedding (Scheduled Manifold Sampling)
        if visual_anchor is not None and blend_weight < 1.0:
            aligned_rep = blend_weight * aligned_rep + (1.0 - blend_weight) * visual_anchor
            
        # 3. Extract last joint telemetry coordinate
        last_qpos = joint_history[:, -1, :self.output_dim]
        
        # 4. Decode future trajectory [B, pred_len, output_dim]
        return self.decoder(aligned_rep, last_qpos)

class InfoNCELoss(nn.Module):
    """
    Bidirectional InfoNCE (Contrastive Loss) Function.
    
    Mathematical Formulation:
      Given a batch of normalized student joint embeddings Z_j in R^(B x D) and
      normalized visual latents V in R^(B x D), we calculate the similarity matrix:
        S = (Z_j @ V^T) / tau
      where tau is the temperature scaling parameter.
      
      The loss maximizes similarity along the diagonal (synchronized pairs) and
      minimizes similarity off-diagonal (mismatched pairs):
        L_j2v = CrossEntropy(S, targets=arange(B))
        L_v2j = CrossEntropy(S^T, targets=arange(B))
        Total Loss = 0.5 * (L_j2v + L_v2j)
        
    This alignment forces the mechanical latent space to mirror the topological
    relations of physical world interactions encoded inside the frozen world model.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, joint_embeddings: torch.Tensor, visual_latents: torch.Tensor) -> torch.Tensor:
        # L2 Normalize embeddings to project them onto a hypersphere
        z_norm = F.normalize(joint_embeddings, p=2, dim=-1)
        v_norm = F.normalize(visual_latents, p=2, dim=-1)
        
        # Calculate cosine similarity matrix
        # Shape: [B, B]
        similarity_matrix = torch.matmul(z_norm, v_norm.t()) / self.temperature
        
        # Target indices are the diagonal (each joint sequence matches its synchronous image frame)
        batch_size = joint_embeddings.size(0)
        targets = torch.arange(batch_size, device=joint_embeddings.device)
        
        # Symmetric Contrastive Loss
        loss_j2v = F.cross_entropy(similarity_matrix, targets)
        loss_v2j = F.cross_entropy(similarity_matrix.t(), targets)
        
        # Mean contrastive accuracy (for metrics logging)
        with torch.no_grad():
            preds = similarity_matrix.argmax(dim=-1)
            acc = (preds == targets).float().mean()
            
        return (loss_j2v + loss_v2j) / 2.0, acc

# Training Step designed for high-throughput execution.
# We define the eager training function first.
def eager_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn_contrastive: nn.Module,
    loss_fn_mse: nn.Module,
    joint_inputs: torch.Tensor,
    action: torch.Tensor,
    visual_latents: torch.Tensor,
    future_qpos_targets: torch.Tensor,
    lambda_weight: float = 0.05,
    blend_weight: float = 1.0
):
    optimizer.zero_grad()
    
    # 1. Project joint history + action to aligned latent space [B, 1024]
    aligned_rep = model.student(joint_inputs, action)
    
    # 2. Extract last qpos and decode future trajectory using the deep sequence decoder with blending
    predictions = model(joint_inputs, action, visual_anchor=visual_latents, blend_weight=blend_weight)
    
    # 3. Compute loss terms (InfoNCE uses the student's raw unblended aligned representation)
    loss_contr, acc = loss_fn_contrastive(aligned_rep, visual_latents)
    loss_mse = loss_fn_mse(predictions, future_qpos_targets)
    
    # Balanced loss: Forecasting_MSE + lambda * InfoNCE_Loss
    total_loss = loss_mse + lambda_weight * loss_contr
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss, loss_contr, loss_mse, acc

def train_contrastive_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    temperature: float = 0.07,
    lambda_weight: float = 0.05,
    target_blend_ratio: float = 0.2,
    schedule_epoch: int = 30,
    device: str = "cuda",
    jitter_enabled: bool = False,
    val_dataloader: DataLoader = None,
    patience: int = 15
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    loss_fn_contrastive = InfoNCELoss(temperature=temperature)
    loss_fn_mse = nn.MSELoss()
    
    best_loss = float('inf')
    best_state_dict = None
    epochs_no_improve = 0
    
    # Attempt to compile the training step for RTX 5080 Blackwell GPU Tensor Cores.
    # If compilation fails (e.g. C compiler missing on Windows), fall back to eager execution.
    use_compile = True
    try:
        train_step_fn = torch.compile(eager_train_step, mode="reduce-overhead")
        print("Model train step wrapped with torch.compile().")
    except Exception as e:
        print(f"Warning: torch.compile initialization failed ({e}). Running in eager mode.")
        train_step_fn = eager_train_step
        use_compile = False
        
    print("Starting Contrastive Alignment + Forecasting training (with Scheduled Manifold Sampling)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_contr = 0.0
        epoch_mse = 0.0
        epoch_acc = 0.0
        
        # Calculate Scheduled Manifold Sampling blend weight for this epoch (Sigmoidal Scheduler)
        if epoch < schedule_epoch:
            blend_weight = 1.0
        else:
            import math
            total_decay_epochs = max(1, epochs - 1 - schedule_epoch)
            z = (epoch - schedule_epoch) / total_decay_epochs
            # Smooth sigmoidal decay shape
            f_z = 1.0 / (1.0 + math.exp(10.0 * (z - 0.5)))
            f_0 = 1.0 / (1.0 + math.exp(-5.0))
            f_1 = 1.0 / (1.0 + math.exp(5.0))
            f_norm = (f_z - f_1) / (f_0 - f_1)
            
            # Clip minimum baseline threshold of the blend ratio at 0.500
            min_ratio = max(0.5, target_blend_ratio)
            blend_weight = min_ratio + (1.0 - min_ratio) * f_norm
            blend_weight = max(min_ratio, min(1.0, blend_weight))
            
        for batch in dataloader:
            joint_inputs = batch["joint_inputs"].to(device)
            visual_latents = batch["visual_latent_anchor"].to(device)
            future_qpos_targets = batch["future_qpos_targets"].to(device)
            
            # Inject localized Gaussian noise into input window during training phase
            if jitter_enabled:
                joint_inputs = joint_inputs + torch.randn_like(joint_inputs) * 0.02
                
            # Next target action vector is the loaded action command from LeRobot
            action = batch["action"].to(device)
            
            try:
                # Try running compiled train step (compilation occurs lazily on first call)
                total_loss, loss_contr, loss_mse, acc = train_step_fn(
                    model=model,
                    optimizer=optimizer,
                    loss_fn_contrastive=loss_fn_contrastive,
                    loss_fn_mse=loss_fn_mse,
                    joint_inputs=joint_inputs,
                    action=action,
                    visual_latents=visual_latents,
                    future_qpos_targets=future_qpos_targets,
                    lambda_weight=lambda_weight,
                    blend_weight=blend_weight
                )
            except Exception as e:
                # Catch lazy compilation failure and fall back permanently to eager mode
                if use_compile:
                    print(f"\n[WARNING] torch.compile failed during execution: {e}.")
                    print("Permanently falling back to high-performance GPU Eager Mode.")
                    train_step_fn = eager_train_step
                    use_compile = False
                    
                    # Execute in eager mode
                    total_loss, loss_contr, loss_mse, acc = train_step_fn(
                        model=model,
                        optimizer=optimizer,
                        loss_fn_contrastive=loss_fn_contrastive,
                        loss_fn_mse=loss_fn_mse,
                        joint_inputs=joint_inputs,
                        action=action,
                        visual_latents=visual_latents,
                        future_qpos_targets=future_qpos_targets,
                        lambda_weight=lambda_weight,
                        blend_weight=blend_weight
                    )
                else:
                    raise e
            
            epoch_loss += total_loss.item() * joint_inputs.size(0)
            epoch_contr += loss_contr.item() * joint_inputs.size(0)
            epoch_mse += loss_mse.item() * joint_inputs.size(0)
            epoch_acc += acc.item() * joint_inputs.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        epoch_contr /= len(dataloader.dataset)
        epoch_mse /= len(dataloader.dataset)
        epoch_acc /= len(dataloader.dataset)
        
        # Validation and early stopping
        val_loss_str = ""
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_loss_mse_accum = 0.0
            val_loss_contr_accum = 0.0
            val_acc_accum = 0.0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    joint_inputs = batch["joint_inputs"].to(device)
                    visual_latents = batch["visual_latent_anchor"].to(device)
                    future_qpos_targets = batch["future_qpos_targets"].to(device)
                    action = batch["action"].to(device)
                    
                    # Compute outputs
                    aligned_rep = model.student(joint_inputs, action)
                    predictions = model(joint_inputs, action, visual_anchor=visual_latents, blend_weight=blend_weight)
                    
                    loss_contr, acc = loss_fn_contrastive(aligned_rep, visual_latents)
                    loss_mse = loss_fn_mse(predictions, future_qpos_targets)
                    total_val_loss = loss_mse + lambda_weight * loss_contr
                    
                    val_loss += total_val_loss.item() * joint_inputs.size(0)
                    val_loss_mse_accum += loss_mse.item() * joint_inputs.size(0)
                    val_loss_contr_accum += loss_contr.item() * joint_inputs.size(0)
                    val_acc_accum += acc.item() * joint_inputs.size(0)
            
            val_loss /= len(val_dataloader.dataset)
            val_loss_mse_accum /= len(val_dataloader.dataset)
            val_loss_contr_accum /= len(val_dataloader.dataset)
            val_acc_accum /= len(val_dataloader.dataset)
            
            val_loss_str = f" | Val Loss: {val_loss:.6f} | Val MSE: {val_loss_mse_accum:.6f} | Val Align Acc: {val_acc_accum * 100:.2f}%"
            
            if val_loss < best_loss:
                best_loss = val_loss
                import copy
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Epoch {epoch+1}/{epochs} | Early stopping contrastive training. Restoring best model with Val Loss: {best_loss:.6f}")
                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                break
        
        # Log epoch summary including current blend weight
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1 or (val_dataloader is not None and epochs_no_improve == 0):
            print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | Total Loss: {epoch_loss:.6f} | InfoNCE: {epoch_contr:.4f} | MSE: {epoch_mse:.6f} | Align Acc: {epoch_acc * 100:.2f}% | Blend Weight: {blend_weight:.3f}{val_loss_str}")
            
        scheduler.step()
        
    return model
        
    return model
