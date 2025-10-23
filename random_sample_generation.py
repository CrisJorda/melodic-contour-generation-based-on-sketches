# Generate the synthetic dataset
# New samples have 3 bars and 2 octave pitch range, in 4/4 time signature

# Imports section
import random
from music21 import stream

# Definition of the path to the results folder
results_folder = "/home/cristina/melodic-contour-generation-based-on-sketches/synthetic_dataset"

# Definition of the constant variables
num_bars = 3
time_signature = "4/4"
pitch_range = (60, 84)  # MIDI numbers: C4 (60) to B5 (83)

# For getting a random key
key_tonic = ["C", "D", "E", "F", "G", "A", "B"]
key_mode = ["major", "minor"]

num_samples = 1

for sample in range(num_samples):
    new_stream = stream.Stream()

    new_part = stream.Part()
    new_part.timeSignature = time_signature
    
    # Generate random key
    tonic = random.choice(tonic)
    mode = random.choice(mode)
    random_key = tonic + mode
    print(random_key)
    new_part.keySignature = random_key

    print(new_stream)