# Sketch-guided Melody Generation

This is the code repository for the article When Lines become Melodies: Sketch-Guided Melody Generation. 

The repository contains all necessary code to reproduce the results obtained in the article.

## Project Structure

```text
.
├── dataset_creation/
│   ├── Irish_data.abc                  # Original Irish folk music dataset (ABC notation)
│   ├── Irish_dataset_generation.py     # Scripts to generate, process & save the Irish dataset
│   └── synthetic_dataset_generation.py # Scripts to generate & save synthetic dataset
├── exp1/
│   ├── config.py     # Experiment 1 configuration
│   ├── dataset.py    # Dataset loading & preprocessing for experiment 1
│   ├── evaluation.py # Metrics & evaluation loop for experiment 1
│   ├── model.py      # Model definition for experiment 1
│   ├── test.py       # Testing script for experiment 1
│   └── train.py      # Training loop for experiment 1
├── exp2/
│   ├── config.py     # Experiment 2 configuration
│   ├── dataset.py    # Dataset loading & preprocessing for experiment 2
│   ├── evaluation.py # Metrics & evaluation loop for experiment 2
│   ├── model.py      # Model definition for experiment 2
│   ├── test.py       # Testing script for experiment 2
│   └── train.py      # Training loop for experiment 2
├── exp3/
│   ├── config.py     # Experiment 3 configuration
│   ├── dataset.py    # Dataset loading & preprocessing for experiment 3
│   ├── evaluation.py # Metrics & evaluation loop for experiment 3
│   ├── model.py      # Model definition for experiment 3
│   ├── test.py       # Testing script for experiment 3
│   └── train.py      # Training loop for experiment 3
├── .gitignore        # Git ignore rules
└── README.md         # Project documentation
