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

- Raw data is stored in: 
data/condition+monitoring+of+hydraulic+systems/ 

- Sensor configuration is defined in: <br/>
config/sensors.json <br/>

# Traditional Machine Learning

Folder: 1_traditional_ml/ <br/>
Notebook: descion_tree_yiling.ipynb <br/>

Approach: <br/>
Feature extraction using tsfresh <br/>
Classification with Decision Tree <br/>
Sensor ablation study to analyze sensor importance <br/>

Files:
PS2_raw_long.csv: raw sensor signal <br/>
PS2_tsfresh_features.csv: extracted features <br/>



