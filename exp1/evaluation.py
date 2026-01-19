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

from dtaidistance import dtw

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
    shuffle=False,
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
checkpoint_path = '/home/cristina/melodic-contour-generation-based-on-sketches/exp1/checkpoints/model_epoch_100.pth'
checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))

model = VariationalAutoencoder(hparams['channels'], hparams['latent_dim'], hparams['vocab_size'], hparams['seq_len'], hparams['frp_len'], hparams['n_layers_encoder'], hparams['embedding_dim'], hparams['n_layers_gru'], hparams['hidden_dim_gru'], hparams['vocab_size_Trp'], hparams['chroma_dim'])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print('Model loaded')

# --------------- Evaluation ----------------
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
    contour = []
    last_pitch = None
    for tok in tokens:
        if tok == 128:  # hold
            if last_pitch is not None:
                contour.append(last_pitch)
        elif tok == 129:  # rest
            continue  # skip rests
        else:
            contour.append(tok)
            last_pitch = tok
    return np.array(contour)

def extract_rhythm_contour(tokens):
    # Compute durations of all notes
    durations = np.zeros(len(tokens))
    element = 0

    while element < len(tokens):
        if tokens[element] < 128: # Note onset
            dur = 1
            next_element = element + 1
            while next_element < len(tokens) and tokens[next_element] == 128:  # Hold
                dur += 1
                next_element += 1
            durations[element] = dur
            element = next_element
        else: # Rest or hold
            element += 1
    #print(durations)

    # Convert durations into density
    density = np.zeros(len(tokens))

    last_density = 0.0
    for i in range(len(tokens)):
        if durations[i] > 0:
            relative_duration = durations[i] / len(tokens)
            last_density = np.log2(1/relative_duration)
        density[i] = last_density
    #print(density)
    
    # 3-level quantization
    quantized_density = np.zeros(len(tokens))
    quantized_density[(density > 0) & (density <= 1.25)] = 0
    quantized_density[(density > 1.25) & (density <= 3.29)] = 1
    quantized_density[density > 3.29] = 2
    #print(quantized_density)

    # Subtract the minimum value
    frr = quantized_density - np.min(quantized_density)

    return frr

def extract_pitch_classes(tokens):
    pitch_classes = []
    for tok in tokens:
        if tok < 128: # ignore holds and rests
            pitch_classes.append(tok % 12)
    return pitch_classes

def estimate_key(pitch_classes):
    # Krumhansl key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Count occurrences of each pitch class
    pitch_class_counts = np.zeros(12)
    for pc in pitch_classes:
        pitch_class_counts[pc] += 1

    if pitch_class_counts.sum() == 0:
        return None, None, set()  # No pitch classes found
    
    pitch_class_counts /= pitch_class_counts.sum()  # Normalize counts
    
    # Compute correlation with each key profile
    max_correlation = -np.inf
    best_tonic = None
    best_mode = None
    for tonic in range(12):
        # Rotate profiles to match tonic
        rotated_major = np.roll(major_profile, tonic)
        rotated_minor = np.roll(minor_profile, tonic)
        
        # Compute correlation
        corr_major = np.corrcoef(pitch_class_counts, rotated_major)[0, 1]
        corr_minor = np.corrcoef(pitch_class_counts, rotated_minor)[0, 1]
        
        # Update best key if needed
        if corr_major > max_correlation:
            max_correlation = corr_major
            best_tonic = tonic
            best_mode = 'major'
        if corr_minor > max_correlation:
            max_correlation = corr_minor
            best_tonic = tonic
            best_mode = 'minor'
    
    # Determine the set of pitch classes in the estimated key
    if best_mode == 'major':
        scale = {0, 2, 4, 5, 7, 9, 11}
    else:
        scale = {0, 2, 3, 5, 7, 8, 10}

    key_pcs = set()
    for interval in scale:
        key_pcs.add((best_tonic + interval) % 12)

    return best_tonic, best_mode, key_pcs

def context_matching(tokens_pred, tokens_context):
    # Extract the pitch classes from the notes
    pitch_classes_pred = extract_pitch_classes(tokens_pred)
    pitch_classes_context = extract_pitch_classes(tokens_context)
    
    # Estimate key using Krumhansl algorithm
    key_tonic, key_mode, key_pcs = estimate_key(pitch_classes_context)
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    best_tonic_name = NOTE_NAMES[key_tonic] if key_tonic is not None else None
    #print(f'Estimated key: {best_tonic_name} {key_mode}')

    if len(pitch_classes_pred) == 0 or len(key_pcs) == 0:
        return 0.0

    # Percentage of notes belonging to the estimated key
    matches = 0
    for pc in pitch_classes_pred:
        if pc in key_pcs:
            matches += 1
    percentage_in_key = int(matches / len(pitch_classes_pred) * 100)
    return percentage_in_key

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

# Define the matching lists
pitch_match_list = []
pitch_r_match_list = []
rhythm_match_list = []
context_match_list = []

#print(len(test_dataloader))

num_eval = 10 # change to 1000

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

        tmidi_true_1 = sample_1[4].long().to(device) # sample_1 if comparing bar_12 with bar_3
        tmidi_true_2 = sample_2[4].long().to(device) # sample_2 if comparing bar_22 with bar_3

        # Generate new bar
        bar3 = model(x_curve, x_context) # tmidi_pred, trp_pred, vchroma_pred, zrp_mu, zrp_logvar, zc_mu, zc_logvar
        tmidi_pred = bar3[0]

        # Save the scores
        first_bar_score = converter.parse(str(sample_1[3][0])).measure(1)
        first_bar_score.write('midi', fp=f"exp1/Midi_Eval/first_bar_score_{i}.mid")

        second_bar_score = converter.parse(str(sample_2[3][0])).measure(2)
        second_bar_score.write('midi', fp=f"exp1/Midi_Eval/second_bar_score_{i}.mid")

        pitch = tmidi_pred[0].argmax(dim=-1).cpu().numpy()
        generated_score = tmidi_to_score(pitch, hparams['seq_len'])
        generated_score.write('midi', fp=f"exp1/Midi_Eval/generated_score_{i}.mid")

        # Get tokens from bars
        tmidi_true_tokens = tmidi_true_2[0]
        print(tmidi_true_tokens)
        tmidi_pred_tokens = tmidi_pred[0].argmax(dim=-1).cpu().numpy()
        print(tmidi_pred_tokens)
        context_tokens = np.concatenate((bar11[0].int(), bar12[0].int())) # all bars from context sample
        #print(context_tokens)

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

        # Compute DTW without rhythm enforcement
        pitch_cost = dtw.distance(tmidi_pred_pitch, tmidi_true_pitch)
        print(pitch_cost)
        pitch_match_list.append(pitch_cost)

        # Compute DTW with rhythm enforcement
        pitch_r_cost = dtw.distance(tmidi_pred_pitch_r, tmidi_true_pitch)
        print(pitch_r_cost)
        pitch_r_match_list.append(pitch_r_cost)

        # Rhythm contour matching
        # Extract contour
        tmidi_pred_rhythm = extract_rhythm_contour(tmidi_pred_tokens)
        #print(tmidi_pred_rhythm)
        tmidi_true_rhythm = extract_rhythm_contour(tmidi_true_tokens)
        #print(tmidi_true_rhythm)

        # Compute DTW
        rhythm_cost = dtw.distance(tmidi_pred_rhythm, tmidi_true_rhythm)
        #print(rhythm_cost)
        rhythm_match_list.append(rhythm_cost)

        # Context matching
        percentage_in_key = context_matching(tmidi_pred_tokens, context_tokens)
        #print(percentage_in_key)
        context_match_list.append(percentage_in_key)

        # Save the results
        #print("First sample")
        #sample_1_score = converter.parse(str(sample_1[3][0])).measure(1)
        #sample_1_score.write('midi', fp=f"Midi_Eval/sample_1_{i}.mid")
        #sample_1_score.show()

        #print("Second sample")
        #sample_2_score = converter.parse(str(sample_2[3][0])).measure(2)
        #sample_2_score.write('midi', fp=f"Midi_Eval/sample_2_{i}.mid")
        #sample_2_score.show()

        #print("Generated melody")
        #generated_score = tmidi_to_score(tmidi_pred_tokens, hparams['seq_len'])
        #generated_score.write('midi', fp=f"Midi_Eval/generated_score_{i}.mid")
        #generated_score.show()                                                                                                                                                   

# Replace these with your actual arrays
pitch_match_list = np.array(pitch_match_list)
pitch_r_match_list = np.array(pitch_r_match_list)
rhythm_match_list = np.array(rhythm_match_list)
context_match_list = np.array(context_match_list)

# Compute mean and standard deviation
results = {
    'pitch without rhythm enforcement': {'mean': np.mean(pitch_match_list), 'std': np.std(pitch_match_list)},
    'pitch with rhythm enforcement': {'mean': np.mean(pitch_r_match_list), 'std': np.std(pitch_r_match_list)},
    'rhythm': {'mean': np.mean(rhythm_match_list), 'std': np.std(rhythm_match_list)},
    'context': {'average': int(np.round(np.mean(context_match_list)))},
}

# Save results to a text file
output_file = os.path.join('exp1', 'alignment_cost_22.txt')
with open(output_file, 'w') as f:
    for key, stats in results.items():
        if key == 'context':
            f.write(f"{key}:\n")
            f.write(f"  Average: {stats['average']}%\n\n")
        else:
            f.write(f"{key}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n\n")

print(f"Results saved to {output_file}")