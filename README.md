# HERGAST

HERGAST: High-resolution Enhanced Relational Graph Attention Network for Spatial Transcriptomics [[paper]](https://doi.org/10.1101/2024.08.09.607422)

This document will help you easily go through the HERGAST model.

![fig1](https://github.com/GYQ-form/HERGAST/assets/79566479/fe08a893-47ac-4fe9-ad25-51f808088748)

## Dependencies

The required Python packages and versions tested in our study are:

```
pytorch==2.1.2
scanpy==1.9.6
scikit-learn==1.3.2
pyg==2.4.0
scipy==1.11.4
numba==0.58.1
```

## Installation

To install the package, run

```bash
git clone https://github.com/GYQ-form/HERGAST.git
cd HERGAST
pip install .
```

## Usage

HERGAST is an approach for spatial clustering and signal amplification in ultra-large-scale and ultra-high-resolution spatial transcriptomics data. HERGAST employs a heterogeneous graph network that integrates gene expression similarity and spatial proximity, incorporating both local and global spatial relationships

## Tutorial

We have prepared several basic tutorials  in https://github.com/GYQ-form/HERGAST/tree/main/Tutorial. You can quickly hands on HERGAST by going through these tutorials.

## Analysis

To enhance the reproducibility of this study, we deposited all the custom code at directory [analysis](https://github.com/GYQ-form/HERGAST/tree/main/analysis) for running HERGAST used in the paper.
