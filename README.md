# Introduction ：
This repository contains the implementation and experimental analysis for the Valve Condition Monitoring in Hydraulic Systems task.
The project investigates both traditional machine learning and deep learning (InceptionTime) approaches for classifying valve conditions based on multivariate sensor time-series data.

# Projekt Structur：

```text
.
├── 0_data_exploration.ipynb
│   └── Initial data analysis and visualization
│
├── 1_traditional_ml/
│   ├── descion_tree_yiling.ipynb
│   │   └── Decision Tree classification with tsfresh features
│   ├── PS2_raw_long.csv
│   │   └── Raw sensor signal (example)
│   └── PS2_tsfresh_features.csv
│       └── Extracted tsfresh features
│
├── 2_deeplearning/
│   ├── inceptiontime_yiling.ipynb
│   │   └── InceptionTime deep learning experiments
│   └── *.keras
│       └── Saved model checkpoints
│
├── config/
│   └── sensors.json
│       └── Sensor configuration and selection
│
├── data/
│   └── condition+monitoring+of+hydraulic+systems/
│       └── Original hydraulic system dataset
│
├── src/
│   └── data_loader.py
│       └── Data loading and preprocessing utilities
│
├── .gitignore
│   └── Git ignore configuration
│
└── README.md
    └── Project documentation

```

# Dataset
The dataset is based on the Condition Monitoring of Hydraulic Systems dataset. <br/>
It consists of multivariate sensor measurements (pressure, temperature, flow, etc.) recorded during hydraulic system operation. <br/>

- Raw data is stored in: <br/>
data/condition+monitoring+of+hydraulic+systems/ <br/>

- Sensor configuration is defined in: <br/>
config/sensors.json <br/>

# Data Exploration (basierend auf code von Herr Boos)
- Notebook: 0_data_exploration.ipynb
- Tasks:
    - Initial inspection of sensor signals
    - Target label distribution
    - Detection of missing values and anomalies
    - Motivation for feature extraction and deep learning approaches

# Traditional Machine Learning
Folder: 1_traditional_ml/ <br/>
Notebook: descion_tree_yiling.ipynb <br/>

- Approach: <br/>
    - Feature extraction using tsfresh <br/>
    - Classification with Decision Tree <br/>
    - Sensor ablation study to analyze sensor importance <br/>

- Files:
    - PS2_raw_long.csv: raw sensor signal <br/>
    - PS2_tsfresh_features.csv: extracted features <br/>

# Deep Learning (InceptionTime)
Folder: 2_deeplearning/
Notebook: inceptiontime_yiling.ipynb

- Model:
    - InceptionTime (aeon library)
    - Multivariate time series input(n_samples, n_channels, n_timepoints)

- Experiments:
    - Multi-split training with different random seeds
    - Sensor ablation study
    - Analysis of accuracy, balanced accuracy and F1-score
Trained model checkpoints are stored as .keras files.

# Implementation Details
- Data loading:
    - Implemented in src/data_loader.py

- Preprocessing:
    - Resampling of all sensor signals to a fixed length
    - Z-normalization per sample and per channel

- Evaluation metrics:
    - Accuracy
    - Balanced Accuracy
    - Macro and Weighted F1-score

# Sensor Ablation Study
- Both traditional ML and deep learning experiments include a sensor ablation study, where one sensor is removed at a time to evaluate its impact on classification performance.

- This helps to:
    - Identify important sensors
    - Analyze robustness of the model
    - Reduce sensor dimensionality

# How to Run
- Create and activate a virtual environment
- Install required dependencies (e.g. aeon, scikit-learn, numpy, pandas)
- Run notebooks in the following order:
    - 0_data_exploration.ipynb
    - 1_traditional_ml/descion_tree_yiling.ipynb
    - 2_deeplearning/inceptiontime_yiling.ipynb

