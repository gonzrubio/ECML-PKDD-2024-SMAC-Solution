# SMAC

Please use the *Forum* in CodaBench during the competition to report issues.

## Dataset

The dataset comprises Sentinel-1 SAR imagery with two multispectral bands (VV/VH). We provide two classification labels (unaffected / affected by earthquake) and a real value representing the earthquake magnitudes for each sample.

Each sample is composed of:
- image with four channels. It contains VV and VH channels for two images at times t0 and t1 (where t0 < t1)
- label contains a binary value (0 for unaffected and 1 for affected area)
- magnitude contains a real value in the range 0-10, representing the magnitude in mb

## Submission

The file *submission.csv* contains a sample submission with the following columns:

- key: unique identifier
- magnitude: predicted magnitude (should be in the range 0-10)
- affected: binary label (0-1)
- flops: resource consumption expressed in FLOPs by PAPI

## Starter Kit

In the *starter-kit* folder, you can find the code to run the baseline using *main.py*.

*requirements.txt* contains the libraries required to run the code.

You can run inference thanks to *inference.py* simply passing your saved checkpoint with *--checkpoint {checkpoint}* to the command line
