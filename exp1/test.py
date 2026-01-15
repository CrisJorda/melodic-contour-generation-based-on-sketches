# Imports section
from config import hparams
from dataset import CombinedDataset
from model import VariationalAutoencoder

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
device = torch.device('cpu')
print(device)

#--------- Definition of Pytorch dataset and Dataloader -------------#
# Definition of the dataset folders
#irish_root = '/home/cristina/melodic-contour-generation-based-on-sketches/irishtunes_dataset'
#synthetic_root = '/home/cristina/melodic-contour-generation-based-on-sketches/synthetics_dataset'
weird_root = '/home/cristina/melodic-contour-generation-based-on-sketches/weird_dataset'

# Create one dataset for all data
combined_dataset = CombinedDataset(weird_root, None)
test_dataset = combined_dataset

# Determine split sizes
#total_size = len(combined_dataset)
#test_size = int(1 * total_size)
#val_size = int(0 * total_size)
#train_size = total_size - test_size - val_size

# Split the dataset

#train_dataset, val_dataset, test_dataset = random_split(
#    combined_dataset, [train_size, val_size, test_size],
#    generator=torch.Generator().manual_seed(42)
#)

# Dataloaders initialization
#train_dataloader = DataLoader(
#    dataset=train_dataset,
#    batch_size=hparams['batch_size'],
#    shuffle=True,
#    drop_last=False,
#)

#val_dataloader = DataLoader(
#    dataset=val_dataset,
#    batch_size=hparams['batch_size'],
#    shuffle=False,
#    drop_last=False,
#)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=4,
    shuffle=False,
    drop_last=False,
)

#print("Train samples:", len(train_dataset))
#print("Validation samples:", len(val_dataset))
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
checkpoint_path = '/home/cristina/melodic-contour-generation-based-on-sketches/checkpoints/model_epoch_100.pth'
checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))

model = VariationalAutoencoder(hparams['channels'], hparams['latent_dim'], hparams['vocab_size'], hparams['seq_len'], hparams['frp_len'], hparams['n_layers_encoder'], hparams['embedding_dim'], hparams['n_layers_gru'], hparams['hidden_dim_gru'], hparams['vocab_size_Trp'], hparams['chroma_dim'])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print('Model loaded')

# Get the summary of the parameters
#summary_model = summary(model, 
#                        input_size=[(hparams['batch_size'], 1, hparams['frp_len']), (hparams['batch_size'], 1, hparams['seq_len_dataset'])],
#                        dtypes=[torch.float, torch.int])
#print(summary_model)

# --------------- Evaluate on test set ---------------
# Set to evaluation mode
"""
model.eval()
start_time=time.time()

# Create empty loss dictionary
test_loss_avg = {"loss_total": 0, "ce_loss_trp": 0, "ce_loss_tmidi": 0, "bce_loss_chroma": 0, "kl_loss": 0}

print('Testing ...')
with torch.no_grad():

    for i, batch in enumerate(test_dataloader):
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

        test_loss_avg["loss_total"] += loss_total.item()
        test_loss_avg["ce_loss_trp"] += ce_loss_trp.item()
        test_loss_avg["ce_loss_tmidi"] += ce_loss_tmidi.item()
        test_loss_avg["bce_loss_chroma"] += bce_loss_chroma.item()
        test_loss_avg["kl_loss"] += kl_loss.item()

    test_loss_avg["loss_total"] /= i
    test_loss_avg["ce_loss_trp"] /= i
    test_loss_avg["ce_loss_tmidi"] /= i
    test_loss_avg["bce_loss_chroma"] /= i
    test_loss_avg["kl_loss"] /= i

    print(
        f"Testing loss: {test_loss_avg['loss_total']:.4f}, "
        f"Relative pitch loss (CE): {test_loss_avg['ce_loss_trp']:.4f}, "
        f"Midi loss (CE): {test_loss_avg['ce_loss_tmidi']:.4f}, "
        f"Chroma vector loss (BCE): {test_loss_avg['bce_loss_chroma']:.4f}, "
        f"VAE loss (KL div): {test_loss_avg['kl_loss']:.4f}"
    )

print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))
"""
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

# Show test scores
bar1_pitches, bar2_pitches, frp, midi_path, tmidi_true, trp_true, vchroma_true = next(iter(test_dataloader)) # 1 batch

model.eval()
model.to(device)
# Use FRP (pitch curve) and bar1_pitch (context) as inputs
x_curve = frp.float().unsqueeze(1).to(device)
x_context = bar1_pitches.int().unsqueeze(1).to(device)
# Apply model
tmidi_pred, trp_pred, _, _, _, _, _ = model(x_curve, x_context)

num_samples = 4

for i in range(num_samples):
    #print("First bar melody")
    first_bar_score = converter.parse(str(midi_path[i])).measure(1)
    first_bar_score.write('midi', fp=f"exp1/Midi_Weird/first_bar_score_{i}.mid")

    #print("Second bar original melody")
    second_bar_score = converter.parse(str(midi_path[i])).measure(2)
    second_bar_score.write('midi', fp=f"exp1/Midi_Weird/second_bar_score_{i}.mid")
    #print("Second bar generated melody")
    pitch = tmidi_pred[i].argmax(dim=-1).cpu().numpy()
    generated_score = tmidi_to_score(pitch, hparams['seq_len'])
    generated_score.write('midi', fp=f"exp1/Midi_Weird/generated_score_{i}.mid")