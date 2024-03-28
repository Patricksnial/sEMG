# Reproduce paper

Code to reproduce [sEMG Gesture Recognition With a Simple Model of Attention](http://proceedings.mlr.press/v136/josephs20a)

# Project Stucture

Data reading and processings are predefined within data.py, including: data relabelling, rectification, butterworth filter, window rolling, and random noise enhancement.


## Reproduce environment

```bash
pip install -r requirements.txt
```

## Run experiment

```bash
python repro.py
```
## Data processing

```bash
python data.py
```
