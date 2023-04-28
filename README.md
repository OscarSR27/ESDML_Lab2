# Micro KWS

Complete flow for keyword spotting on microcontrollers. From data collection to data preparation to training and deployment.

## Context
This project is used as part of the lab accompanying the lecture: Embedded System Design for Machine Learning offered by EDA@TUM.

## Information on Virtual Environemts

We recommend to use Python version 3.8 because this is the only version the assignment was tested on!

To keep the disk usage in the lab diretcories low, please try to use only one virtual python environment for all lab assignments (when possible).

Init virtual environment:

```bash
# Run inside top level directory of cloned micro-kws repository
virtualenv -p python3.8 venv  # Alternative: python3.8 -m venv venv
```

Make sure to enter this virtual environment before starting working on the lab exercises:

```bash
source venv/bin/activate
```

To leave the environment, just type `deactivate`.

For each lab, there exist `requirements.txt` fils with all the Python dependencies. Feel free to install them now to save time later. It might take a lot of time depending on you internet connection.

```bash
# Lab 1
pip install -r 1_train/requirements.txt

# Lab 2
pip install -r 2_deploy/requirements.txt
pip install -r 3_run/requirements.txt
pip install -r 4_debug/requirements.txt  # optional
```

## Structure of this repository
The following directories can be found at the top level of this repository:
- `0_record/`: Provides utilities for recording and preprocessing new dataset samples (Optional)
- `1_train/`: Contains MicroKWS training flow and tutorial (Lab 1)
- `2_deploy/`: Contains a tutorial for generating MicroKWS kernels for a pre-trained model using the TVM Framework (Lab 2, Part 1)
- `3_run/`: Provides target software demo for deploying the MicroKWS application to a microcontroller (Lab 2, Part 2)
- `4_debug/`: Contains a python tool to debug the target application running on the device (Lab 2, optional)
- ~~`5_bench/`: Contains examples on how to benchmark TinyML models efficiently (optional)~~
