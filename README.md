# ClearClimate - Workshop 4

## XAI models for climate prediction, impact assessment, and decision support

This workshop (WS4) will introduce participants to the fundamentals of Explainable Artificial Intelligence (XAI) and its relevance to climate science. It will begin with a primer on XAI, highlighting why transparency, interpretability, and trust are critical when using AI in scientific and decision-making contexts. A focused literature review will showcase how XAI has been applied in climate-related research, offering insight into current practices and identifying key challenges. Through hands-on exercises, participants will train deep learning models for spatial downscaling (DeepESD and U-Net) and apply XAI techniques such as saliency maps and Integrated Gradients (via Captum and Quantus) to interpret the model’s outputs. The session will explore how these tools enhance understanding and confidence in AI-driven climate applications. The workshop will conclude with an open discussion on insights gained and future directions.

---

## 💻 Environment Installation Instructions

We recommend using **conda** or **mamba** to manage the environment.

### Step 1: Install Miniconda (if needed)

* Download from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Follow installation instructions for your OS

### Step 2: Create and activate the environment using `requirements.txt`

```
# Create new environment and install requirements
conda create -n clearclimate-ws4 python=3.11 -y
conda activate clearclimate-ws4
pip install -r requirements.txt
```

Optionally, use `mamba` for faster installation:

```
mamba create -n clearclimate-ws4 python=3.11 -y
mamba activate clearclimate-ws4
pip install -r requirements.txt
```

## 📦 Data Preparation

The dataset used in this workshop consists of ERA5 (low-resolution) and CERRA (high-resolution) temperature fields. The data has been subsetted for practical training and evaluation during the workshop.

### Files provided:

* `train_era5.nc`, `train_cerra.nc`: training input/output (4000 random samples)
* `val_era5.nc`, `val_cerra.nc`: validation input/output (1000 random samples)
* `test_era5.nc`, `test_cerra.nc`: full year 2019 input/output for model testing

These NetCDF files preserve temporal, latitudinal, and longitudinal dimensions.

Due to size limitations, the full dataset is not hosted in the GitHub repository. You can download the `data/` folder from the following link:

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
