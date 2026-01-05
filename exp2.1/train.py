# Imports section
from config import hparams
from dataset import CombinedDataset
from model import VariationalAutoencoder

import os
import time

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import matplotlib.pyplot as plt

# Seed definition
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)

# Work on GPU if it is available in the machine, otherwise run on CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

#--------- Definition of Pytorch dataset and Dataloader -------------#

# Definition of the dataset folders
irish_root = '/home/cristina/melodic-contour-generation-based-on-sketches/irishtunes_dataset'
synthetic_root = '/home/cristina/melodic-contour-generation-based-on-sketches/synthetics_dataset'

# Create one dataset for all data
combined_dataset = CombinedDataset(irish_root, synthetic_root)

# Determine split sizes
total_size = len(combined_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    combined_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Dataloaders initialization
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=hparams['batch_size'],
    shuffle=True,
    drop_last=False,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=hparams['batch_size'],
    shuffle=False,
    drop_last=False,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=hparams['batch_size'],
    shuffle=False,
    drop_last=False,
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ------------------- Training -----------------------------
# Function for extracting ground truth vectors
def compute_true_vectors(bar2_pitches):
    """
    Obtain Tmidi true and Trp true from the ground truth pitch vector of the 2nd bar

    Args:
        bar2_pitches: [B, 48] of MIDI pitches (0 means rest)

    Returns:
        tmidi_true: [B, 24] tokens (0-127 = notes, 128 = hold, 129 = rest)
        trp_true: [B, 24] quantized relative pitch curve'
    """

    # Define constants from hparams
    batch = bar2_pitches.size(0)
    seq_len_dataset = hparams['seq_len_dataset']
    seq_len = hparams['seq_len']

    # Define grid beat ticks
    beat_ticks = [0, 1/4, 1/3, 1/2, 2/3, 3/4] # 6 uneven ticks per beat -> 24 ticks per measure

    # Define quantization bins
    bins = np.array([-9, -5, -1, 0, 1, 5, 9], dtype=np.float32)

    # Define empty vectors
    tmidi_true = torch.zeros(batch, seq_len, dtype=torch.float32)
    trp_true = torch.zeros(batch, seq_len, dtype=torch.float32)
    vchroma_true = torch.zeros(batch, 12, dtype=torch.float32)

    # Create new grid
    # - In seq_len_dataset = 48, the grid is even, so 12 steps per beat (48/4)
    # - Map the 6 uneven ticks into 12 uniform steps
    tick_indices = np.zeros_like(beat_ticks) # = [0, 3, 4, 6, 8, 9]
    for t in range(len(beat_ticks)):
        tick_indices[t] = round(beat_ticks[t]*12)

    # - Create grid
    grid = [] # = [0, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 21, 24, 27, 28, 30, 32, 33, 36, 39, 40, 42, 44, 45]
    for beat in range(4):
        base = beat * 12
        for tick in tick_indices:
            grid += [int(base + tick)]

    for b in range(batch):
        pitches = bar2_pitches[b] # has seq_len_dataset

        # Compute vchroma_true
        # Get pitch classes
        pitch_classes = np.unique(pitches % 12)
        # Set active classes to 1
        vchroma_true[b, pitch_classes] = 1.0

        # Map bar2_pitches to uneven 24-positions grid
        pitches = pitches[grid] # has seq_len

        # Compute Tmidi_true
        # Go through the pitch vector of each batch
        for t in range(seq_len):
            if pitches[t] == 0:
                tmidi_true[b,t] = 129 # rest condition
            elif t>0 and pitches[t] == pitches[t-1] and pitches[t] != 0:
                tmidi_true[b,t] = 128 # hold condition
            else:
                tmidi_true[b,t] = pitches[t] # MIDI pitch number for onset position
        
        # Compute Trp_true
        # Get the onsets vector
        onsets = np.where((tmidi_true[b] >= 0) & (tmidi_true[b] <= 127))[0]

        if len(onsets) >= 2:
            for pos in range(len(onsets) - 1):
                # For every pair of consecutive onsets in positions i and j = i + n where n > 0, 
                # Calculate the MIDI interval Δ = Tmidi[j] - Tmidi[i]
                i = onsets[pos].item()
                j = onsets[pos + 1].item()
                delta = tmidi_true[b, j] - tmidi_true[b, i]
                n = j - i

                # Then perform a linear interpolation between i + 1 and j (k points), resulting in Trp[k]
                for k in range(i+1, j+1):
                    trp_true[b, k] = delta * ((k-i)/n)
        
        # Quantize Trp_true
        trp_quantizated = torch.zeros(batch, seq_len, dtype=torch.float32)
        bins_tensor = torch.tensor(bins) # convert to tensor so it can be comptatible with trp_true
        for k in range(seq_len):
            diff = np.abs(bins_tensor - trp_true[b, k])
            nearest_bin_idx = np.argmin(diff)
            trp_quantizated[b, k] = nearest_bin_idx
    
    return tmidi_true, trp_quantizated, vchroma_true

# Define loss function
def loss_function(trp_pred, trp_true, tmidi_pred, tmidi_true,
                  lambda_trp, lambda_tmidi,
                  vchroma_pred, vchroma_true,
                  lambda_chroma,
                  zrp_mu, zrp_logvar, zc_mu, zc_logvar, variational_beta):
    
    # Flatten batch + time before computing cross-entropies
    B, T, Vtrp  = trp_pred.shape
    _, _, Vmidi = tmidi_pred.shape

    trp_pred_flat   = trp_pred.reshape(B*T, Vtrp)
    trp_true_flat   = trp_true.reshape(B*T)
    tmidi_pred_flat = tmidi_pred.reshape(B*T, Vmidi)
    tmidi_true_flat = tmidi_true.reshape(B*T)

    # Compute cross-entropy losses
    ce = nn.CrossEntropyLoss()
    ce_loss_trp   = ce(trp_pred_flat, trp_true_flat) * lambda_trp
    ce_loss_tmidi = ce(tmidi_pred_flat, tmidi_true_flat) * lambda_tmidi

    # Compute binary cross-entropy loss
    bce = nn.BCEWithLogitsLoss()
    bce_loss_chroma = lambda_chroma * bce(vchroma_pred, vchroma_true.float())

    # Compute KL divergence loss
    kl_loss_zrp = -0.5 * torch.mean(1 + zrp_logvar - zrp_mu.pow(2) - zrp_logvar.exp())
    kl_loss_zc = -0.5 * torch.mean(1 + zc_logvar - zc_mu.pow(2) - zc_logvar.exp())
    kl_total = variational_beta * (kl_loss_zrp + kl_loss_zc)

    # Weighted sum
    loss_total = ce_loss_trp + ce_loss_tmidi + bce_loss_chroma + kl_total

    return loss_total, ce_loss_trp, ce_loss_tmidi, bce_loss_chroma, kl_total

model = VariationalAutoencoder(hparams['channels'], hparams['latent_dim'], hparams['vocab_size'], hparams['seq_len'], hparams['frp_len'], hparams['n_layers_encoder'], hparams['embedding_dim'], hparams['n_layers_gru'], hparams['hidden_dim_gru'], hparams['vocab_size_Trp'], hparams['chroma_dim'])
model = model.to(device)
optimizer = optim.Adam(params=model.parameters(), lr=hparams['learning_rate'])

# Get the summary of the parameters
summary_model = summary(model, 
                        input_size=[(hparams['batch_size'], 1, hparams['frp_len']), (hparams['batch_size'], 1, hparams['seq_len_dataset'])],
                        dtypes=[torch.float, torch.int])
print(summary_model)

# Save checkpoint every 10 epochs
checkpoints_dir = '/home/cristina/melodic-contour-generation-based-on-sketches/checkpoints'
os.makedirs(checkpoints_dir, exist_ok=True)

# Early stopping settings
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Training loop
start_time=time.time()

# Set to training mode
model.train()

# Create empty loss dictionary
train_loss_avg = {"loss_total": [], "ce_loss_trp": [], "ce_loss_tmidi": [], "bce_loss_chroma": [], "kl_loss": []}
val_loss_avg = {"loss_total": [], "ce_loss_trp": [], "ce_loss_tmidi": [], "bce_loss_chroma": [], "kl_loss": []}

print('Training ...')
for epoch in range(hparams['num_epochs']):
    train_loss_avg["loss_total"].append(0)
    train_loss_avg["ce_loss_trp"].append(0)
    train_loss_avg["ce_loss_tmidi"].append(0)
    train_loss_avg["bce_loss_chroma"].append(0)
    train_loss_avg["kl_loss"].append(0)

    for i, batch in enumerate(train_dataloader):
        print(i)
        # Unpack dataset output
        bar1_pitches, bar2_pitches, frp, midi_path = batch
        
        # Use FRP (pitch curve) and bar1_pitch (context) as inputs
        x_curve = frp.float().unsqueeze(1).to(device)
        x_context = bar1_pitches.int().unsqueeze(1).to(device)

        # Use bar2_pitches to get ground truth vectors
        tmidi_true, trp_true, vchroma_true = compute_true_vectors(bar2_pitches)
        trp_true = trp_true.long().to(device)
        tmidi_true = tmidi_true.long().to(device)
        vchroma_true = vchroma_true.long().to(device)

        # Forward pass
        tmidi_pred, trp_pred, vchroma_pred, zrp_mu, zrp_logvar, zc_mu, zc_logvar = model(x_curve, x_context)  # [B, seq_len, vocab_size]
        loss_total, ce_loss_trp, ce_loss_tmidi, bce_loss_chroma, kl_loss = loss_function(trp_pred, trp_true, tmidi_pred, tmidi_true, hparams['lambda_trp'], hparams['lambda_tmidi'], vchroma_pred, vchroma_true, hparams['lambda_chroma'], zrp_mu, zrp_logvar, zc_mu, zc_logvar, hparams['variational_beta'])

        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()

        optimizer.step()

        train_loss_avg["loss_total"][-1] += loss_total.item()
        train_loss_avg["ce_loss_trp"][-1] += ce_loss_trp.item()
        train_loss_avg["ce_loss_tmidi"][-1] += ce_loss_tmidi.item()
        train_loss_avg["bce_loss_chroma"][-1] += bce_loss_chroma.item()
        train_loss_avg["kl_loss"][-1] += kl_loss.item()

    train_loss_avg["loss_total"][-1] /= i
    train_loss_avg["ce_loss_trp"][-1] /= i
    train_loss_avg["ce_loss_tmidi"][-1] /= i
    train_loss_avg["bce_loss_chroma"][-1] /= i
    train_loss_avg["kl_loss"][-1] /= i

    print(
        f"Epoch [{epoch+1} / {hparams['num_epochs']}], " 
        f"Training loss: {train_loss_avg['loss_total'][-1]:.4f}, "
        f"Relative pitch loss (CE): {train_loss_avg['ce_loss_trp'][-1]:.4f}, "
        f"Midi loss (CE): {train_loss_avg['ce_loss_tmidi'][-1]:.4f}, "
        f"Chroma vector loss (BCE): {train_loss_avg['bce_loss_chroma'][-1]:.4f}, "
        f"VAE loss (KL div): {train_loss_avg['kl_loss'][-1]:.4f}"
    )

    # Set to validation mode
    model.eval()
    with torch.no_grad():
        val_loss_avg["loss_total"].append(0)
        val_loss_avg["ce_loss_trp"].append(0)
        val_loss_avg["ce_loss_tmidi"].append(0)
        val_loss_avg["bce_loss_chroma"].append(0)
        val_loss_avg["kl_loss"].append(0)

        for i, batch in enumerate(val_dataloader):
            # Unpack dataset output
            bar1_pitches, bar2_pitches, frp, midi_path = batch
            
            # Use FRP (pitch curve) and bar1_pitch (context) as inputs
            x_curve = frp.float().unsqueeze(1).to(device)
            x_context = bar1_pitches.int().unsqueeze(1).to(device)

            # Use bar2_pitches to get ground truth vectors
            tmidi_true, trp_true, vchroma_true = compute_true_vectors(bar2_pitches)
            trp_true = trp_true.long().to(device)
            tmidi_true = tmidi_true.long().to(device)
            vchroma_true = vchroma_true.long().to(device)

            # Forward pass
            tmidi_pred, trp_pred, vchroma_pred, zrp_mu, zrp_logvar, zc_mu, zc_logvar = model(x_curve, x_context)  # [B, seq_len, vocab_size]
            loss_total, ce_loss_trp, ce_loss_tmidi, bce_loss_chroma, kl_loss = loss_function(trp_pred, trp_true, tmidi_pred, tmidi_true, hparams['lambda_trp'], hparams['lambda_tmidi'], vchroma_pred, vchroma_true, hparams['lambda_chroma'], zrp_mu, zrp_logvar, zc_mu, zc_logvar, hparams['variational_beta'])

            val_loss_avg["loss_total"][-1] += loss_total.item()
            val_loss_avg["ce_loss_trp"][-1] += ce_loss_trp.item()
            val_loss_avg["ce_loss_tmidi"][-1] += ce_loss_tmidi.item()
            val_loss_avg["bce_loss_chroma"][-1] += bce_loss_chroma.item()
            val_loss_avg["kl_loss"][-1] += kl_loss.item()

        val_loss_avg["loss_total"][-1] /= i
        val_loss_avg["ce_loss_trp"][-1] /= i
        val_loss_avg["ce_loss_tmidi"][-1] /= i
        val_loss_avg["bce_loss_chroma"][-1] /= i
        val_loss_avg["kl_loss"][-1] /= i

        print(
            f"Epoch [{epoch+1} / {hparams['num_epochs']}], " 
            f"Validation loss: {val_loss_avg['loss_total'][-1]:.4f}, "
            f"Relative pitch loss (CE): {val_loss_avg['ce_loss_trp'][-1]:.4f}, "
            f"Midi loss (CE): {val_loss_avg['ce_loss_tmidi'][-1]:.4f}, "
            f"Chroma vector loss (BCE): {val_loss_avg['bce_loss_chroma'][-1]:.4f}, "
            f"VAE loss (KL div): {val_loss_avg['kl_loss'][-1]:.4f}"
        )
        
    model.train()

    # Save the model every 10 epochs
    if (epoch + 1) % 10 == 0:       
        save_path = os.path.join(      
            checkpoints_dir, f"model_epoch_{epoch+1}.pth"
        )
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hparams": hparams
        }, save_path)
        print(f"Saved checkpoint: {save_path}")
    
    # Early stopping
    current_val_loss = val_loss_avg['loss_total'][-1]

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        epochs_no_improve = 0

        # Save best model
        best_path = os.path.join(checkpoints_dir, "best_model.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hparams": hparams
        }, best_path)
    
    else:
        epochs_no_improve += 1
    
    # Check patience
    if epochs_no_improve >= hparams['patience_early_stopping']:
        early_stop = True
        break

print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

# Plot the loss curves
# Total loss curve
plt.figure(figsize=(8, 6))

train_loss, = plt.plot(train_loss_avg['loss_total'], label="Train loss")
val_loss, = plt.plot(val_loss_avg['loss_total'], label="Validation loss")

plt.xlabel("Epochs")
plt.ylabel("Total loss")
plt.grid(True)
plt.legend()

plt.savefig("total_loss.png")
plt.close()

# Other losses curves
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

loss_types = ["ce_loss_trp", "ce_loss_tmidi", "bce_loss_chroma", "kl_loss"]
y_labels = ["Relative pitch loss (CE)", "Midi loss (CE)", "Chroma vector loss (BCE", "VAE loss (KL div)"]

for i, loss_type in enumerate(loss_types):
    (train_h,) = axs[i].plot(train_loss_avg[loss_type], label="Train")
    (val_h,) = axs[i].plot(val_loss_avg[loss_type], label="Validation")
    axs[i].set_xlabel("Epochs")
    axs[i].set_title(y_labels[i])
    axs[i].grid(True)

fig.legend(handles=[train_h, val_h], loc="upper center", bbox_to_anchor=(0.5, 1.15))
plt.savefig("other_loss.png")
plt.close()