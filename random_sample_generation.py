# Generate the synthetic dataset
# New samples have 3 bars and 2 octave pitch range, in 4/4 time signature

# Imports section
import random
from music21 import stream, meter, key

# Definition of the path to the results folder
results_folder = "synthetic_dataset"

# Definition of the constant variables
num_bars = 3
time_signature = meter.TimeSignature('4/4')
pitch_range = (60, 84)  # MIDI numbers: C4 (60) to B5 (83)

# For getting a random key
keys = ["C", "D", "E", "F", "G", "A", "B", "c", "d", "e", "f", "g", "a", "b"]

num_samples = 1

for sample in range(num_samples):
    new_stream = stream.Stream()

    new_part = stream.Part()
    new_part.timeSignature = time_signature
    
    # Generate random key
    random_key = key.Key(random.choice(keys))
    print(random_key)
    new_part.keySignature = random_key

    new_stream.show()