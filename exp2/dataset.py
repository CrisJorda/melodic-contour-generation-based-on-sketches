# Imports section
from config import hparams

import os
import numpy as np

import torch
from torch.utils.data import Dataset

from music21 import converter, note
from scipy.signal import butter, filtfilt
from scipy.interpolate import make_lsq_spline

#--------- Definition of Pytorch dataset and Dataloader -------------#

# Extract the pitches from 1 bar
def extract_pitches_1bar(notes):
    seq_len = hparams['seq_len_dataset']
    measeure_length = 4 # time signature is 4/4, so 4 beats per measure

    pitch_vector = np.zeros(seq_len, dtype=np.float32)

    onsets = []
    pitches = []

    for n in notes:
        if isinstance(n, note.Note):
            onsets.append(n.offset)
            pitches.append(n.pitch.midi)

    # Convert to numpy arrays
    onsets = np.array(onsets, dtype=np.float32)
    pitches = np.array(pitches, dtype=np.int16)

    # Convert note onset (offset in beats) to a grid position [0..47]
    norm_onsets = np.mod(onsets, measeure_length) / measeure_length
    quant_onsets = np.round(norm_onsets * (seq_len - 1)).astype(int)

    # Fill the pitch_vector
    # Sustain each note until the next onset
    for i, onset in enumerate(quant_onsets):
        if i + 1 < len(quant_onsets):
            end = quant_onsets[i + 1]
        else:
            end = seq_len
        pitch_vector[onset:end] = pitches[i]
    
    return pitch_vector, quant_onsets, pitches

# Compute the relative pitch function for 2nd bar
def compute_frp(onsets, pitches):
    # Definition of the constant variables
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

    # Return the function
    return frp.astype(np.float32)

def compute_spline_derivative(frp):
    # Define the spline parameters
    x = np.linspace(0, 1, len(frp))
    y = frp
    control_points = 12 # Number of control points
    k = 3  # Spline degree
    num_internal_knots = control_points - k - 1 # 12-3-1 = 8 internal knots
    t_internal = np.linspace(0, 1, num_internal_knots + 2)[1:-1]  # Internal knots vector
    # Make the knot vector (k+1)-regular by adding boundary knots
    t = np.r_[(x[0],)*(k+1),
            t_internal,
            (x[-1],)*(k+1)]

    # Fit the spline
    spline = make_lsq_spline(x, y, t, k)

    # Compute the derivative of the spline
    xs = np.linspace(0, 1, hparams['frp_len'])
    spline_derivative = spline.derivative(1)(xs)

    return spline_derivative

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
    seq_len_dataset = hparams['seq_len_dataset']
    seq_len = hparams['seq_len']

    # Define grid beat ticks
    beat_ticks = [0, 1/4, 1/3, 1/2, 2/3, 3/4] # 6 uneven ticks per beat -> 24 ticks per measure

    # Define quantization bins
    bins = np.array([-9, -5, -1, 0, 1, 5, 9], dtype=np.float32)

    # Define empty vectors
    tmidi_true = torch.zeros(seq_len, dtype=torch.float32)
    trp_true = torch.zeros(seq_len, dtype=torch.float32)
    vchroma_true = torch.zeros(12, dtype=torch.float32)

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

    pitches = bar2_pitches[bar2_pitches > 0] # has seq_len_dataset

    # Compute vchroma_true
    # Get pitch classes
    pitch_classes = np.unique(pitches % 12)
    # Set active classes to 1
    vchroma_true[pitch_classes] = 1.0

    # Map bar2_pitches to uneven 24-positions grid
    pitches = bar2_pitches[grid] # has seq_len

    # Compute Tmidi_true
    # Go through the pitch vector of each batch
    for t in range(seq_len):
        if pitches[t] == 0:
            tmidi_true[t] = 129 # rest condition
        elif t>0 and pitches[t] == pitches[t-1] and pitches[t] != 0:
            tmidi_true[t] = 128 # hold condition
        else:
            tmidi_true[t] = int(pitches[t]) # MIDI pitch number for onset position
    
    # Compute Trp_true
    # Get the onsets vector
    onsets = np.where((tmidi_true >= 0) & (tmidi_true <= 127))[0]
    if len(onsets) >= 2:
        for pos in range(len(onsets) - 1):
            # For every pair of consecutive onsets in positions i and j = i + n where n > 0, 
            # Calculate the MIDI interval Δ = Tmidi[j] - Tmidi[i]
            i = onsets[pos].item()
            j = onsets[pos + 1].item()
            delta = tmidi_true[j] - tmidi_true[i]
            n = j - i

            # Then perform a linear interpolation between i + 1 and j (k points), resulting in Trp[k]
            for k in range(i+1, j+1):
                trp_true[k] = delta * ((k-i)/n)
    
    # Quantize Trp_true
    trp_quantizated = torch.zeros(seq_len, dtype=torch.float32)
    bins_tensor = torch.tensor(bins) # convert to tensor so it can be comptatible with trp_true
    for k in range(seq_len):
        diff = np.abs(bins_tensor - trp_true[k])
        nearest_bin_idx = np.argmin(diff)
        trp_quantizated[k] = nearest_bin_idx
    
    return tmidi_true, trp_quantizated, vchroma_true

# Process the 2 bars
def process_bars(midi_path):
    score = converter.parse(midi_path)

    # Extract 1st and 2nd bar from score
    bar1 = score.measure(1).flatten().notes
    bar2 = score.measure(2).flatten().notes

    # Convert bars into 48-step pitch vectors
    bar1_pitch_vector, _, _ = extract_pitches_1bar(bar1)
    bar2_pitch_vector, bar2_onsets, bar2_pitches = extract_pitches_1bar(bar2)

    # Compute the FRP for 2nd bar
    frp = compute_frp(bar2_onsets, bar2_pitches)

    # Compute the spline representation of the FRP
    spline_derivative = compute_spline_derivative(frp)

    # Convert all vectors to numpy arrays
    bar1_pitch_vector = np.array(bar1_pitch_vector)
    bar2_pitch_vector = np.array(bar2_pitch_vector)
    spline_derivative = np.array(spline_derivative)

    # Compute the true vectors from 2nd bar
    tmidi_true, trp_true, vchroma_true = compute_true_vectors(bar2_pitch_vector)

    # Return pitch vectors as numpy arrays
    return bar1_pitch_vector, bar2_pitch_vector, spline_derivative, tmidi_true, trp_true, vchroma_true

# Definition the Pytorch Dataset
class CombinedDataset(Dataset):
    """
    This dataset combines samples from Irish Folk Music Dataset and the synthetic dataset.

    Returns for each sample:
    - past_pitches (from 1st bar, encoder input, len 48)
    - curr_pitches (from 2nd bar, decoder target, len 48)
    - frp (from 2nd bar, pitch curve, len 512)
    - midi_path
    """
    def __init__(self, irish_root, synthetic_root, seq_len=hparams['seq_len_dataset'], frp_len=hparams['frp_len']):
        self.seq_len = seq_len
        self.frp_len = frp_len

        # Collect all MIDI files from both datasets
        self.files = []

        # Irish dataset
        irish_files = []
        max_irish_files = 455766  # Limit to 455,766 files as Experiment 1
        for f in os.listdir(irish_root):
            if f.endswith(".mid") or f.endswith(".midi"):
                irish_files.append(os.path.join(irish_root, f))

        # Synthetic dataset
        synthetic_files = []
        for f in os.listdir(synthetic_root):
            if f.endswith(".mid") or f.endswith(".midi"):
                synthetic_files.append(os.path.join(synthetic_root, f))

        self.files.extend(irish_files[:max_irish_files])
        self.files.extend(synthetic_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        midi_path = self.files[idx]

        # Extract pitches from both bars and frp from 2nd bar
        past_pitches, curr_pitches, spline_derivative, tmidi_true, trp_true, vchroma_true = process_bars(midi_path)

        return past_pitches, curr_pitches, spline_derivative, midi_path, tmidi_true, trp_true, vchroma_true