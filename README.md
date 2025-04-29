# HERGAST
[![DOI](https://zenodo.org/badge/817575211.svg)](https://doi.org/10.5281/zenodo.15000094)

HERGAST: High-resolution Enhanced Relational Graph Attention Network for Spatial Transcriptomics [[paper]](https://www.nature.com/articles/s41467-025-59139-w)

This document will help you easily go through the HERGAST model.

![fig1](https://github.com/GYQ-form/HERGAST/assets/79566479/fe08a893-47ac-4fe9-ad25-51f808088748)

## Dependencies

The required Python packages and versions tested in our study are:

```
pytorch==2.4.1
scanpy==1.10.3
scikit-learn==1.5.2
pyg==2.6.1
scipy==1.14.1
numpy==2.0.1
pandas==2.2.3
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

## Reference

If you find our work useful in your research or if you use parts of this code, please consider citing our [paper](https://www.nature.com/articles/s41467-025-59139-w):

Gong, Y., Yuan, X., Jiao, Q. *et al.* Unveiling fine-scale spatial structures and amplifying gene expression signals in ultra-large ST slices with HERGAST. *Nat Commun* **16**, 3977 (2025). https://doi.org/10.1038/s41467-025-59139-w
