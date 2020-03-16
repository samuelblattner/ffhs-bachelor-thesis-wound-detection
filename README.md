# Bachelor Thesis

This is the Wound Detection Suite I used to train and test deep neural networks for my Bachelor Thesis.
It's entirely experimental, so, use at your own risk.

## Prerequisites

- Python >= 3.6
- apt-get install python3-venv


## Setup

To setup the Suite, simply run:

    $ . ./setup.sh
    
from the project's root directory.


## Simple detection example

1. Download the state-of-the-art checkpoint [here](https://drive.google.com/file/d/1bUhnPgfFH868NNcB11m_lWb2vXhUtdKo/view?usp=sharing)
2. To use the checkpoint for wound detection on a given image, run:
    
        $ python main.py retina-resnet152 --weights <path-to-checkpoint-file> --out_dir <path-to-output-dir> detect <path-to-image-file> 

Detections will be stored to <path-to-output-dir>.