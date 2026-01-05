# Imports section
from config import hparams

import os
import numpy as np

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

    # Return pitch vectors as numpy arrays
    return np.array(bar1_pitch_vector), np.array(bar2_pitch_vector), np.array(spline_derivative)

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
        for root in [irish_root, synthetic_root]:
            for f in os.listdir(root):
                if f.endswith(".mid") or f.endswith(".midi"):
                    self.files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        midi_path = self.files[idx]

        # Extract pitches from both bars and frp from 2nd bar
        past_pitches, curr_pitches, frp = process_bars(midi_path)

        return past_pitches, curr_pitches, frp, midi_path