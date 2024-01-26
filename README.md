# MFRS_ENSAE

## Overview
This repository contains the Jupyter Notebook and Python files for our Movie Face Recognition System. We developped a Siamese Network with Triplet Loss in order to predict who is the actor on the screen.

## Installation
To run this project:
1. Clone the repository: `git clone https://github.com/AlexisReve/MFRS_ENSAE.git`
2. Navigate to the repository directory: `cd MFRS_ENSAE`

## Data 

In order to augment the volume of our initial data, we use the following dataset : http://vis-www.cs.umass.edu/lfw/

## Information relative to files

The reader woud be particularly interested about the following files :

- `face_recognition/test_model.ipynb`.
- `face_recognition/training_model.ipynb`.
- `face_recognition/utils.py` which gathers the main functions used in other files.

The `face_detection` folder is for now useless and consists of preliminary test of yolov-face model.
