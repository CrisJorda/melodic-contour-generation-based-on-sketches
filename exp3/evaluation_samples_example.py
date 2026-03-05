# Import section
from config import hparams
from dataset import CombinedDataset
from model import VariationalAutoencoder

import os

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from music21 import converter, note, stream, meter
from scipy.signal import butter, filtfilt

from dtaidistance import dtw

import matplotlib.pyplot as plt

# Seed definition
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)

# Work on GPU if it is available in the machine, otherwise run on CPU
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
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
    batch_size=1,
    shuffle=True,
    drop_last=False,
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

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
checkpoint_path = '/home/cristina/melodic-contour-generation-based-on-sketches/exp3/checkpoints_8/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))

model = VariationalAutoencoder(hparams['channels'], hparams['latent_dim'], hparams['vocab_size'], hparams['seq_len'], hparams['spline_derivative_len'], hparams['n_layers_encoder'], hparams['embedding_dim'], hparams['n_layers_gru'], hparams['hidden_dim_gru_encoder'], hparams['hidden_dim_gru_decoder'], hparams['vocab_size_Trp'], hparams['chroma_dim'])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print('Model loaded')

def extract_true_rhythm(true_tokens, pred_tokens):
    # Extract true rhythm from true tokens
    true_rhythm = []
    for tok in true_tokens:
        if tok == 128:  # hold
            true_rhythm.append(1)
        elif tok == 129:  # rest
            true_rhythm.append(2)
        else: # onset
            true_rhythm.append(0)
    
    # Apply rhythm enforcement to predicted tokens
    tokens_r = []
    last_pitch = None

    for i, rhythm in enumerate(true_rhythm):
        if rhythm == 0:  # onset
            if pred_tokens[i] < 128: # pred_tokens has valid onset in i
                tokens_r.append(pred_tokens[i])
                last_pitch = pred_tokens[i]
            elif last_pitch is not None: # pred_tokens has hold or rest, use last valid pitch
                tokens_r.append(last_pitch)
            else: # no valid pitch yet, use rest
                tokens_r.append(129)
        elif rhythm == 1:  # hold
            tokens_r.append(128)
        elif rhythm == 2:  # rest
            tokens_r.append(129)

    return tokens_r

def extract_pitch_contour(tokens):
    pitches = []
    onsets = []
    
    # Extract contours
    for i, tok in enumerate(tokens):
        if tok < 128:  # onset
            pitches.append(tok)
            onsets.append(i)
    
    onsets = np.array(onsets, dtype=np.float32)
    pitches = np.array(pitches, dtype=np.int16)

    # Compute the relative pitch function
    frp_len = hparams['frp_len']
    filter_cutoff = 1/50
    filter_order = 5

    if len(onsets) == 0 or len(pitches) == 0:
        # Check empty arrays
        frp = np.zeros(frp_len, dtype=np.float32)

    else:
        # Piecewise-linear interpolation
        x = np.linspace(onsets.min(), onsets.max(), frp_len)
        y = np.interp(x, onsets, pitches)

        # Low-pass Butterworth filtering
        # to smooth the linear interpolation
        b, a = butter(filter_order, filter_cutoff, btype='low')
        y_smooth = filtfilt(b, a, y)

        # Normalization
        # Find midpoint in the curve range
        pitch_range = y_smooth.max() - y_smooth.min()
        middle_val = y_smooth.min() + pitch_range/2

        # Substract midpoint and divide by 12
        # 12 because 12 semitones = 1 octave
        frp = (y_smooth - middle_val) / 12.0

    return frp.astype(np.float32)

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

num_eval = 20

loader_iter = iter(test_dataloader)

model.eval()

with torch.no_grad():
    for i in range(num_eval):
        print(i)

        sample_1 = next(loader_iter)
        sample_2 = next(loader_iter)

        # Each sample has: bar1_pitches, bar2_pitches, frp, midi_path, tmidi_true, trp_true, vchroma_true

        # Extract useful information
        bar11 = sample_1[0] # bar1_pitches
        x_context = bar11.int().unsqueeze(1).to(device)
        bar12 = sample_1[1] # bar2_pitches

        bar22 = sample_2[2] # frp
        x_curve = bar22.float().unsqueeze(1).to(device)

        tmidi_true_1 = sample_1[4].long().to(device) # sample_1 if comparing bar_12 with bar_3 -> reference
        tmidi_true_2 = sample_2[4].long().to(device) # sample_2 if comparing bar_22 with bar_3 -> related

        # Generate new bar
        bar3 = model(x_curve, x_context) # tmidi_pred, trp_pred, vchroma_pred, zrp_mu, zrp_logvar, zc_mu, zc_logvar
        tmidi_pred = bar3[0]

        # Save the scores
        first_bar_score = converter.parse(str(sample_1[3][0])).measure(1)
        #first_bar_score.write('midi', fp=f"exp3/Midi_Eval_8/first_bar_score_{i}.mid")
        #first_bar_score.show()

        second_bar_score = converter.parse(str(sample_2[3][0])).measure(2)
        #second_bar_score.write('midi', fp=f"exp3/Midi_Eval_8/second_bar_score_{i}.mid")
        #second_bar_score.show()

        pitch = tmidi_pred[0].argmax(dim=-1).cpu().numpy()
        generated_score = tmidi_to_score(pitch, hparams['seq_len'])
        #generated_score.write('midi', fp=f"exp3/Midi_Eval_8/generated_score_{i}.mid")
        #generated_score.show()

        # Get tokens from bars
        tmidi_true_tokens = tmidi_true_2[0]
        #print(tmidi_true_tokens)
        tmidi_pred_tokens = tmidi_pred[0].argmax(dim=-1).cpu().numpy()
        #print(tmidi_pred_tokens)

        context = np.concatenate((bar11[0].int(), bar12[0].int())) # all bars from context sample
        #print(context)

        # Pitch contour matching
        # Apply rhythm enforcement
        tmidi_pred_tokens_r = extract_true_rhythm(tmidi_true_tokens, tmidi_pred_tokens)

        # Extract contour
        tmidi_pred_pitch = extract_pitch_contour(tmidi_pred_tokens)
        #print(tmidi_pred_pitch)
        tmidi_pred_pitch_r = extract_pitch_contour(tmidi_pred_tokens_r)
        #print(tmidi_pred_pitch_r)
        tmidi_true_pitch = extract_pitch_contour(tmidi_true_tokens)
        #print(tmidi_true_pitch)

        # Save the plot with the 3 frps
        color_true = '#C5860A' #"#DDB22F"
        color_pred = "#8E7DBE" #'#6FA8DC' 
        color_pred_r = '#7FB77E' # '#1B7A46'

        plt.figure(figsize=(10, 3))

        plt.plot(tmidi_true_pitch, label="Ground truth", linewidth=3, color='tab:cyan')
        plt.plot(tmidi_pred_pitch, label="Predicted", linestyle="--", linewidth=2, color='tab:orange')
        plt.plot(tmidi_pred_pitch_r, label="Predicted (rhythm enforced)", linestyle="--", linewidth=2, color='tab:purple')

        plt.title("Relative pitch function comparison", fontsize=11)
        plt.ylabel("Normalized pitch", fontsize=10)
        plt.xlabel("Normalized time", fontsize=10)

        plt.ylim(-1, 1)

        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        #plt.show()
        plt.savefig(f"exp3/Midi_Eval_8/plot_{i}.svg", dpi=150)
        plt.close()