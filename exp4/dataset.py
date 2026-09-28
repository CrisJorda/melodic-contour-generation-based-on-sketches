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
    xs = np.linspace(0, 1, hparams['spline_derivative_len'])
    spline_derivative = spline.derivative(1)(xs)

    return spline_derivative

# Extract the onsets and pitches from 1 bar
def extract_onsets_and_pitches(notes):
    seq_len = hparams['seq_len_dataset']
    measeure_length = 4 # time signature is 4/4, so 4 beats per measure

    raw_onsets = []
    raw_pitches = []

    for n in notes:
        if isinstance(n, note.Note):
            raw_onsets.append(float(n.offset))
            raw_pitches.append(n.pitch.midi)

    # Convert to numpy arrays
    raw_onsets = np.array(raw_onsets, dtype=np.float32)
    raw_pitches = np.array(raw_pitches, dtype=np.int16)

    # Convert note onset (offset in beats) to a grid position [0..47]
    norm_onsets = np.mod(raw_onsets, measeure_length) / measeure_length
    grid_onsets = np.round(norm_onsets * (seq_len - 1)).astype(int)

    # Quantize to 16-step grid (3 steps per beat)
    step = 3
    max_onset = seq_len - step  # 45 for step=3
    quant_onsets = (np.round(grid_onsets / step) * step).astype(int)
    quant_onsets = np.clip(quant_onsets, 0, max_onset)

    # Sort by onset and deduplicate collisions
    order = np.argsort(quant_onsets)
    quant_onsets = quant_onsets[order]
    quant_pitches = raw_pitches[order]

    unique_onsets = []
    unique_pitches = []
    last_onset = None
    for onset, pitch in zip(quant_onsets, quant_pitches):
        if last_onset is not None and onset == last_onset:
            # If there's a collision, drop this extra note
            continue
        unique_onsets.append(onset)
        unique_pitches.append(pitch)
        last_onset = onset
    
    return np.array(unique_onsets, dtype=np.int64), np.array(unique_pitches, dtype=np.int64)

def compute_pitch_vector(onsets, pitches):
    seq_len = hparams['seq_len_dataset']

    pitch_vector = np.zeros(seq_len, dtype=np.float32)

    # Fill the pitch_vector
    # Sustain each note until the next onset
    for i, onset in enumerate(onsets):
        if i + 1 < len(onsets):
            end = onsets[i + 1]
        else:
            end = seq_len
        pitch_vector[onset:end] = pitches[i]

    return pitch_vector

# Function for extracting ground truth vectors
def compute_true_vectors(onsets, pitches):
    """
    Obtain Tmidi true and Trp true from the ground truth pitch vector of the 2nd bar

    Args:
        onsets: [N] array of note onsets
        pitches: [N] array of note pitches

    Returns:
        tmidi_true: [B, 48] tokens (0-127 = notes, 128 = hold, 129 = rest)
        trp_true: [B, 48] quantized relative pitch curve'
    """

    # Define constants from hparams
    seq_len = hparams['seq_len']

    # Define empty vectors
    tmidi_true = torch.full((seq_len,), 129, dtype=torch.long) # Initialize with rest condition
    trp_true = torch.zeros(seq_len, dtype=torch.long)
    vchroma_true = torch.zeros(12, dtype=torch.float32)

    pitches = np.array(pitches, dtype=np.int64)
    onsets = np.array(onsets, dtype=np.int64)

    if len(onsets) == 0 or len(pitches) == 0:
        # If there are no onsets or pitches, return the initialized vectors
        return tmidi_true, trp_true, vchroma_true

    # Compute vchroma_true
    # Get pitch classes
    pitch_classes = np.unique(pitches % 12)
    # Set active classes to 1
    vchroma_true[pitch_classes] = 1.0

    # Compute Tmidi_true
    # Go through the pitch vector of each batch
    for i, (onset, pitch) in enumerate(zip(onsets, pitches)):
        if onset < 0 or onset >= seq_len:
            continue  # Skip invalid onsets

        tmidi_true[onset] = int(pitch)  # MIDI pitch number for onset position

        if i+1 < len(onsets):
            end = min(onsets[i+1], seq_len)  # Ensure we don't go out of bounds
        else:
            end = seq_len

        if onset + 1 < end:
            # Fill the sustain (hold) condition for the duration of the note
            tmidi_true[onset + 1:end] = 128  # Hold condition
    
    # Compute Trp_true
    # Define quantization bins
    bins = torch.tensor([-9, -5, -1, 0, 1, 5, 9], dtype=torch.float32)
    if len(onsets) >= 2:
        for pos in range(len(onsets) - 1):
            # For every pair of consecutive onsets in positions i and j = i + n where n > 0, 
            # Calculate the MIDI interval Δ = Tmidi[j] - Tmidi[i]
            i = int(onsets[pos])
            j = int(onsets[pos + 1])

            if j<= i:
                continue  # Skip if j is not greater than i

            delta = tmidi_true[j] - tmidi_true[i]
            n = j - i

            for k in range(i+1, j+1):
                # Linear interpolation between i + 1 and j (k points)
                relative_value = delta * ((k-i)/n)
                # Quantize the relative value to the nearest bin
                diff = torch.abs(bins - relative_value)
                nearest_bin_idx = torch.argmin(diff)
                trp_true[k] = nearest_bin_idx
    
    return tmidi_true, trp_true, vchroma_true

# Process the 2 bars
def process_bars(midi_path):
    score = converter.parse(midi_path)

    # Extract 1st and 2nd bar from score
    bar1 = score.measure(1).flatten().notes
    bar2 = score.measure(2).flatten().notes

    # Extract onsets and pitches for both bars
    bar1_onsets, bar1_pitches = extract_onsets_and_pitches(bar1)
    bar2_onsets, bar2_pitches = extract_onsets_and_pitches(bar2)

    # Compute the FRP for 2nd bar
    frp = compute_frp(bar2_onsets, bar2_pitches)

    # Compute the spline representation of the FRP
    spline_derivative = compute_spline_derivative(frp)
    spline_derivative = np.array(spline_derivative)

    # Convert bars into 48-step pitch vectors
    bar1_pitch_vector = compute_pitch_vector(bar1_onsets, bar1_pitches)
    bar2_pitch_vector = compute_pitch_vector(bar2_onsets, bar2_pitches)
    bar1_pitch_vector = np.asarray(bar1_pitch_vector)
    bar2_pitch_vector = np.asarray(bar2_pitch_vector)

    # Compute the true vectors from 2nd bar
    tmidi_true, trp_true, vchroma_true = compute_true_vectors(bar2_onsets, bar2_pitches)

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
    def __init__(self, midi_files, seq_len=hparams['seq_len_dataset'], frp_len=hparams['frp_len']):
        self.seq_len = seq_len
        self.frp_len = frp_len

        self.files = midi_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        midi_path = self.files[idx]

        # Extract pitches from both bars and frp from 2nd bar
        past_pitches, curr_pitches, spline_derivative, tmidi_true, trp_true, vchroma_true = process_bars(midi_path)

        return past_pitches, curr_pitches, spline_derivative, midi_path, tmidi_true, trp_true, vchroma_true