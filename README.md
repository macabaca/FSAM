# FSAM (Frequency-Segmented Adaptive Modeling for Multivariate Time Series Forecasting)

A time series forecasting framework with frequency-segmented adaptive mechanisms for improved prediction accuracy.

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)

## Overview

FSAM (Frequency-Segmented Adaptive Modeling for Multivariate Time Series Forecasting) is a state-of-the-art time series forecasting model that leverages frequency-segmented adaptive mechanisms to capture temporal dependencies at different frequency scales. The model is designed to handle various multivariate time series datasets including electricity consumption, weather data, and traffic flow.

## Environment Setup

### Prerequisites
- Python 3.8
- CUDA-compatible GPU (recommended for training)
- Linux environment

### Installation

1. **Create a conda environment**
   ```bash
   conda create -n fsam_env python=3.8
   conda activate fsam_env
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

To run all experiments with the best configurations:

```bash
bash scripts/FSAM/all.sh
```

### Individual Dataset Experiments

You can also run experiments on specific datasets:

```bash
# Electricity dataset
bash scripts/FSAM/elec_best.sh

# Weather dataset
bash scripts/FSAM/weather_best.sh

# Traffic dataset
bash scripts/FSAM/traffic_best.sh

# Solar dataset
bash scripts/FSAM/solar_best.sh

# ETTh1 dataset
bash scripts/FSAM/etth1_best.sh

# ETTh2 dataset
bash scripts/FSAM/etth2_best.sh

# ETTm1 dataset
bash scripts/FSAM/ettm1_best.sh

# ETTm2 dataset
bash scripts/FSAM/ettm2_best.sh
```

## Dataset

The project supports multiple time series datasets:

- **ECL**: Electricity consumption data
- **ETT**: Electricity Transformer Temperature (ETTh1, ETTh2, ETTm1, ETTm2)
- **Weather**: Weather forecasting data
- **Solar**: Solar energy production data
- **Traffic**: Traffic flow data

Datasets should be placed in the `dataset/` directory with the following structure:
```
dataset/
├── ECL/
│   └── electricity.csv
├── ETT/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── Weather/
│   └── weather.csv
├── Solar/
│   └── solar_AL.csv
└── Traffic/
    └── traffic.csv
```