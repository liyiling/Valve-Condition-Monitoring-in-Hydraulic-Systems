# 

# Projekt Struktur：
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


#

