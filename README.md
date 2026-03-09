# Curve2Melody: Sketch-Guided Melody Generation

This is the code repository for the article Curve2Melody: Sketch-Guided Melody Generation, which contains all necessary code to reproduce the results obtained in the article.

Curve2Melody is a multimodal AI generative system that enables to create new melodies by providing a hand-drawn sketch along with a contextual previous melody. Building on the "Draw and Listen" music inpainting framework [1], we redesigned the input structure and representation, and consequently optimized the architecture to obtain a substantially lighter model while maintaining the ability to recursively generate stylistically coherent melodies. We introduced a novel methodology for representing melodic contour curves by approximating the sketch with a cubic spline and using its first-order derivative directly as model input. 

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
## Installation
It is recommended to use a virtual environment and install the dependencies from `requirements.txt`.

## Dataset Generation
The dataset used to train and evaluate the Curve2Melody model is the combination of Irish folk and synthetic samples.
The original Irish folk music dataset in ABC notation is located at: `dataset_creation/Irish_data.abc`.
Run the scripts `Irish_dataset_generation.py` and `synthetic_dataset_generation.py` from the folder `dataset_creation` to obtain the combined dataset.

## Running the experiments
Each experiment folder (`exp1, exp2, exp3`) contains its own configuration, dataset preprocessing, model, training, testing and evaluation scripts.

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

Experiment 3 corresponds to the proposed Curve2Melody system, which achieves strong performance with a substantially lighter architecture than the other experiments and the ‘Draw and Listen’ [1] baseline.

## References
[1] C. Benetatos and Z. Duan, “Draw and listen! a sketch-based system for music inpainting,” in Transactions of the International Society for Music Information Retrieval, 2022, pp. 141–155.

## Cite
If you use this code or data in your research, please cite:
to be done once published

