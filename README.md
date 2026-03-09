# Sketch-Guided Melody Generation

This is the code repository for the article When Lines become Melodies: Sketch-Guided Melody Generation. 

The repository contains all necessary code to reproduce the results obtained in the article.

## Project Structure

```text
.
├── dataset_creation/
│   ├── Irish_data.abc                    # Original Irish folk music dataset (ABC notation)
│   ├── Irish_dataset_generation.py       # Scripts to generate, process & save the Irish dataset
│   └── synthetic_dataset_generation.py   # Scripts to generate & save synthetic dataset
├── exp1/
│   ├── config.py                         # Experiment 1 configuration
│   ├── dataset.py                        # Dataset loading & preprocessing for experiment 1
│   ├── evaluation.py                     # Metrics & evaluation loop for experiment 1
│   ├── model.py                          # Model definition for experiment 1
│   ├── test.py                           # Testing script for experiment 1
│   └── train.py                          # Training loop for experiment 1
├── exp2/
│   ├── config.py                         # Experiment 2 configuration
│   ├── dataset.py                        # Dataset loading & preprocessing for experiment 2
│   ├── evaluation.py                     # Metrics & evaluation loop for experiment 2
│   ├── model.py                          # Model definition for experiment 2
│   ├── test.py                           # Testing script for experiment 2
│   └── train.py                          # Training loop for experiment 2
├── exp3/
│   ├── config.py                         # Experiment 3 configuration
│   ├── dataset.py                        # Dataset loading & preprocessing for experiment 3
│   ├── evaluation.py                     # Metrics & evaluation loop for experiment 3
│   ├── model.py                          # Model definition for experiment 3
│   ├── test.py                           # Testing script for experiment 3
│   └── train.py                          # Training loop for experiment 3
├── .gitignore                            # Git ignore rules
└── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies

```

## Dataset Generation
Run the scripts `Irish_dataset_generation.py` and `synthetic_dataset_generation.py` from the folder `dataset_creation`.

## Experiment configurations
Although the three experiment folders share the same file names, each experiment has distinct configurations and implementations. The main differences are summarized in the next table.

| Trainable Parameter              | Experiment 1              | Experiment 2         | Experiment 3         |
|----------------------------------|---------------------------|----------------------|----------------------|
| Contour representation           | Relative pitch function   | Spline derivative    | Spline derivative    |
| `spline_derivative_len`         | –                         | 512                  | 256                  |
| `n_layers_encoder`              | 7                         | 7                    | 2                    |
| `hidden_dim_gru_encoder`        | 2048                      | 2048                 | 512                  |
| `hidden_dim_gru_decoder`        | 2048                      | 2048                 | 512                  |
| Total trainable parameters       | 581,220,741               | 581,220,741          | 17,571,973           |

