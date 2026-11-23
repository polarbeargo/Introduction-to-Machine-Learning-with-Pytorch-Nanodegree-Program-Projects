# Introduction to Machine Learning with PyTorch

This repository contains three machine-learning projects covering unsupervised
learning, image classification with a convolutional neural network, and
supervised learning for donor identification.

## Projects

### 1. Identify Customer Segments

Use demographic data from the general German population and mail-order customers
to discover population segments and compare them with the company's customer
base. The notebook covers missing-data analysis, feature selection and
encoding, scaling, principal component analysis (PCA), and clustering.

- [Project notebook](<./%20Identify%20Customer%20Segments/Identify_Customer_Segments.ipynb>)
- [Rendered notebook](<./%20Identify%20Customer%20Segments/Identify_Customer_Segments.html>)
- [Feature data dictionary](<./%20Identify%20Customer%20Segments/Data_Dictionary.md>)

The project data includes population and customer demographic subsets, plus a
feature summary file. See the notebook and data dictionary for details.

### 2. CIFAR-10 Image Classifier

Train and evaluate a convolutional neural network to classify 32 × 32 color
images into the 10 CIFAR-10 categories. The notebook includes training-only
image augmentation, a validation split, model training, and a final test-set
evaluation.

- [Project notebook](<./Create%20Your%20Own%20Image%20Classifier%202.0/CIFAR-10_Image_Classifier-STARTER.ipynb>)
- [Saved model weights](<./Create%20Your%20Own%20Image%20Classifier%202.0/artifacts/cifar10_best_weights.pth>)
- [Run summary and metrics](<./Create%20Your%20Own%20Image%20Classifier%202.0/artifacts/cifar10_run_summary.json>)

The saved run reports **86.40% test accuracy** and **87.32% best validation
accuracy**. The notebook's comparison with a hypothetical object-detection
benchmark is only an initial reference: image classification does not perform
object detection or localization, so it is not a direct product comparison.

### 3. Finding Donors for CharityML

Explore census data and train supervised-learning models to identify people
whose income is above $50,000, as a proxy for finding likely donors. The notebook
covers preprocessing, a naive predictor baseline, model evaluation and tuning,
feature importance, and feature selection.

- [Project notebook](./Finding_Donors_Project/finding_donors.ipynb)
- [Project data](./Finding_Donors_Project/census.csv)
- [Rendered report](./Finding_Donors_Project/report.html)
- [Project description](./Finding_Donors_Project/project_description.md)

## Getting started

1. Install Python and Jupyter Notebook or JupyterLab.
2. Install the libraries needed for the project you want to run. The projects
   use common data-science libraries such as NumPy, pandas, scikit-learn, and
   Matplotlib. The image-classification notebook additionally requires PyTorch
   and torchvision.
3. From the repository root, start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open the notebook linked above and run its cells in order. The CIFAR-10
   notebook may download the dataset on its first run and can use CUDA, Apple
   MPS, or CPU depending on the available hardware.

Each project is self-contained in its own folder. Refer to its notebook and
supporting files for project-specific details and data requirements.
