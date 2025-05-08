# ClearClimate - Workshop 4

## XAI models for climate prediction, impact assessment, and decision support

This workshop (WS4) will introduce participants to the fundamentals of Explainable Artificial Intelligence (XAI) and its relevance to climate science. It will begin with a primer on XAI, highlighting why transparency, interpretability, and trust are critical when using AI in scientific and decision-making contexts. A focused literature review will showcase how XAI has been applied in climate-related research, offering insight into current practices and identifying key challenges. Through hands-on exercises, participants will train a basic deep learning model for spatial downscaling and apply simple XAI techniques such as saliency maps and SHAP to interpret the model’s outputs. The interactive session will provide an opportunity to explore how these tools can enhance understanding and confidence in AI-driven climate applications. The workshop will close with an open discussion to reflect on insights gained and outline next steps for the upcoming activities.

---

## 💻 Environment Installation Instructions

We recommend using **conda** or **mamba** to manage the environment.

### Step 1: Install Miniconda (if needed)

- Download from: https://docs.conda.io/en/latest/miniconda.html
- Follow installation instructions for your OS

### Step 2 (Optional): Install Mamba for faster dependency resolution

```
conda install mamba -n base -c conda-forge
```

### Step 3: Create and activate the environment

```
mamba env create -f environment.yml  # or conda env create -f environment.yml
conda activate clearclimate-ws4
```

## Data Preparation

The dataset used in this workshop consists of ERA5 (low-resolution) and CERRA (high-resolution) temperature fields. The data has been subsetted for practical training and evaluation during the workshop.

### Files provided:

- `train_era5.nc`, `train_cerra.nc`: training input/output (4000 random samples)
- `val_era5.nc`, `val_cerra.nc`: validation input/output (1000 random samples)
- `test_era5.nc`, `test_cerra.nc`: full year 2019 input/output for model testing

These NetCDF files preserve temporal, latitudinal, and longitudinal dimensions.

Due to size limitations, the full dataset is not hosted in the GitHub repository.
You can download the `data/` folder from the following link:

👉 [Download data folder from Predictia Cloud](https://cloud.predictia.es/s/wXD3TAqb39W2X8Y)

After downloading, place the folder inside the repository root:

```
mv ~/Downloads/data ./clear-climate-ws4/
```

### 📁 Folder Structure (after preprocessing)
```
data/
├── train_era5.nc       # 4000 samples of ERA5 temperature data (0.25º)
├── train_cerra.nc      # Corresponding 4000 samples of CERRA temperature data (0.05º)
├── val_era5.nc         # 1000 validation samples from ERA5
├── val_cerra.nc        # Corresponding 1000 validation samples from CERRA
├── test_era5.nc        # Full year 2019 ERA5 data
└── test_cerra.nc       # Full year 2019 CERRA data
```