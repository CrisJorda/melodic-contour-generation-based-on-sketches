# Import section
from config import hparams
from dataset import CombinedDataset
from model import VariationalAutoencoder

import os
import time

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from music21 import converter, note, stream, meter

# Seed definition
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)

# Work on GPU if it is available in the machine, otherwise run on CPU
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'
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

# --------------- Load trained model -----------------
checkpoint_path = '/home/cristina/melodic-contour-generation-based-on-sketches/checkpoints/model_epoch_100.pth'
checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))

model = VariationalAutoencoder(hparams['channels'], hparams['latent_dim'], hparams['vocab_size'], hparams['seq_len'], hparams['frp_len'], hparams['n_layers_encoder'], hparams['embedding_dim'], hparams['n_layers_gru'], hparams['hidden_dim_gru'], hparams['vocab_size_Trp'], hparams['chroma_dim'])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print('Model loaded')

# --------------- Evaluation ----------------

def tmidi_to_score(tmidi_tokens, seq_len):
    score = stream.Stream()
    score.append(meter.TimeSignature('4/4'))
    tick_duration = 4.0 / seq_len

    # Definition of previous element
    prev_element = None

    # Go through every element of tmidi
    for element in tmidi_tokens:
        if 1 <= element <= 127:
            # MIDI pitch onset
            n = note.Note(int(element))
            n.quarterLength = tick_duration
            score.append(n)
            prev_element = n

        elif element == 128:
            # Hold token
            if prev_element is not None:
                prev_element.quarterLength += tick_duration
            # else ignore invalid continuation

        elif element == 129:
            # Rest token
            r = note.Rest()
            r.quarterLength = tick_duration
            score.append(r)
            prev_element = r

        else:
            # silently skip invalid tokens, including tok=0
            pass

    return score

num_eval = 1 # change to 1000

loader_iter = iter(test_dataloader)

model.eval()

with torch.no_grad():
    for i in range(num_eval):
        sample_1 = next(loader_iter) # bar1_pitches, bar2_pitches, frp, midi_path
        sample_2 = next(loader_iter) # bar1_pitches, bar2_pitches, frp, midi_path

        # Extract useful information
        bar11 = sample_1[0] # bar1_pitches
        x_context = bar11.int().unsqueeze(1).to(device)

        bar22 = sample_2[2] # frp
        x_curve = sample_2[2].float().unsqueeze(1).to(device)

        #tmidi_true, _, _ = compute_true_vectors(bar22)
        #tmidi_true = tmidi_true.long().to(device)

        # Generate new bar
        bar3 = model(x_curve, x_context) # tmidi_pred, trp_pred, vchroma_pred, zrp_mu, zrp_logvar, zc_mu, zc_logvar
        tmidi_pred = bar3[0] 

        # Save the results
        #print("First sample")
        sample_1_score = converter.parse(str(sample_1[3][i]))
        sample_1_score.write('midi', fp=f"exp1/Midi_Eval/sample_1_{i}.mid")

        #print("Second sample")
        sample_2_score = converter.parse(str(sample_2[3][i]))
        sample_2_score.write('midi', fp=f"exp1/Midi_Eval/sample_2_{i}.mid")

        #print("Generated melody")
        pitch = tmidi_pred[i].argmax(dim=-1).cpu().numpy()
        generated_score = tmidi_to_score(pitch, hparams['seq_len'])
        generated_score.write('midi', fp=f"exp1/Midi_Eval/generated_score_{i}.mid")                                                                                    
