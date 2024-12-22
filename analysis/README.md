# HERGAST-Analysis

To enhance the reproducibility of this study, this directory contains all the custom code of running HERGAST used in the analysis in the paper. Here we briefly introduce the function of each script to make this repo more accessible to users.

- Analysis of the Visium HD and Xenium data has been formated as two tutorials [here](https://github.com/GYQ-form/HERGAST/tree/main/Tutorial).
- simulation_clustering
  - generate_data.py : script to generate simulation data using the Human Lung Cell Atlas (HLCA) based on the spatial pattern (also have been deposited here as png image) and save as h5ad anndata.
  - HERGAST_pipeline.py : runing HERGAST pipeline in simulated data and record the resource consumption of each step.
  - leiden_resolution.py : sensitivity analysis of the resolution parameter in Leiden clustering, applied in simulated data.
  - subunit_test.py : record the resource consumption and model performance vary with the number of splitting patches in DIC. 
- simulation_amplification
  - reconstruct_all.py : generate spatial gene expression (based on the pattern given by png images here), add noise, reconstruct the expression using HERGAST and evaluate results.
- SMI_Lung
  - HERGAST_pipeline.py : runing HERGAST pipeline in SMI Lung cancer dataset.
  - PCA_dim.py : running HERGAST with different number of PCs as input and record the results using the SMI Lung cancer dataset.
  - leiden_resolution.py : sensitivity analysis of the resolution parameter in Leiden clustering, applied in the SMI Lung cancer dataset.
- plot : this sub-directory contains additional code to reproduce figures in the paper which has not been included in the above scripts.
  - simu_res.R : Visualize the performance of spatial clustering of different methods in simulation study.
  - time_memory.R : Visualize the  results scalability comparison (time and memory consumption) on simulated data.  
  - SMI_metric.R : Visualize the performance of spatial clustering of different methods in SMI Lung cancer.
  - sparse_ablation.R : Visualize the results of ablation study on different data conditions. 
  - leiden_res.R : generate the line plot and heatmap plot in Leiden resolution parameter sensitivity analysis.
  - num_patches.R : line plot of influence of patch division on resource consumption and performance.  
  - PCA_dim.R : Visualize influence of PCA dimensions on HERGAST model performance.
  - data : contains result data used to create plots


