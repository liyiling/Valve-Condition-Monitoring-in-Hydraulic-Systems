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
    
- Ergebniss von Ablation des Desicion Tree:
    - ohne PS5,EPS1,PS4,FS1,VS1 bekommen wir sogar bessere Ergebnisse,es bedeutet, die 5 Sensoren werden bei Desicion Tree Training entfernt.
    - ohne PS2 bekommen wir die schlechste Ergebnisse, es bedeutet, PS2 ist am wichtigsten.
    - Danach habe ich PS2 und eine andere Sensor von den übrigen 11 Sensoren Kombination gemacht, zeigt PS2-TS2-Kombination ist am besten.
    - Aber die Optimierung der PS2-TS2-Kombination ist nicht signifikant, also p wert ist größer als 0.05.
    - Zum Schluss habe ich PS2 ausgewählt als der einzige Sensor zum Desicion Tree Trainieren.
```text
| removed_sensor | mean_bal_acc | delta_vs_baseline | std_bal_acc |
|----------------|--------------|-------------------|-------------|
| PS2            | 0.816481     | -0.093148         | 0.013351    |
| PS3            | 0.898750     | -0.010880         | 0.007616    |
| PS6            | 0.900046     | -0.009583         | 0.009697    |
| TS3            | 0.900370     | -0.009259         | 0.003019    |
| CE             | 0.900787     | -0.008843         | 0.006187    |
| FS2            | 0.901111     | -0.008519         | 0.013560    |
| PS1            | 0.901389     | -0.008241         | 0.013269    |
| TS1            | 0.902315     | -0.007315         | 0.011111    |
| TS2            | 0.902731     | -0.006898         | 0.003451    |
| SE             | 0.905185     | -0.004444         | 0.020470    |
| TS4            | 0.905694     | -0.003935         | 0.006324    |
| CP             | 0.908102     | -0.001528         | 0.005932    |
| VS1            | 0.909630     | 0.000000          | 0.007882    |
| None (baseline)| 0.909630     | 0.000000          | NaN         |
| PS5            | 0.909954     | 0.000324          | 0.019872    |
| EPS1           | 0.911111     | 0.001481          | 0.007641    |
| PS4            | 0.914259     | 0.004630          | 0.003072    |
| FS1            | 0.936759     | 0.027130          | 0.029910    |

```

- Ergebniss von Ablation des Inceptiontime:

```text
```

# How to Run
- Create and activate a virtual environment
- Install required dependencies (e.g. aeon, scikit-learn, numpy, pandas)
- Run notebooks in the following order:
    - 0_data_exploration.ipynb
    - 1_traditional_ml/descion_tree_yiling.ipynb
    - 2_deeplearning/inceptiontime_yiling.ipynb

# Decision Tree – Experimental Conclusions
- Sensor Importance
    - Sensor PS2 ist der wichtigste Sensor im gesamten Projekt.
    - Bereits die Verwendung von nur PS2 ist ausreichend, um ein gut performendes Decision-Tree-Modell zu trainieren.
Mit PS2 und 9 ausgewählten deskriptiven statistischen Merkmalen erzielt der Decision Tree die beste Leistung. Die durchschnittliche Genauigkeit liegt bei ca. 97 %.

- Ergebnisse mit tsfresh
    - Der Einsatz von tsfresh liefert in diesem Projekt schwächere Ergebnisse als erwartet, selbst wenn nur der wichtigste Sensor PS2 verwendet wird.
    - Vorgehensweise mit tsfresh:
          1. Es wurde nur PS2 für die Merkmalsextraktion mit tsfresh verwendet, wodurch zunächst 777 Features generiert wurden.
          2. Anschließend wurden notwendige Vorverarbeitungsschritte durchgeführt:
              - Behandlung invalider Werte
              - Entfernen von Features ohne Varianz→ dadurch blieben 426 Features übrig.
              - Danach wurden dreimalige Train-Test-Splits sowie eine Normalisierung (Scaling) durchgeführt.

- Ergebnisse:
    - Ohne Feature Selection, aber mit normalisierten tsfresh-Features, wurde eine durchschnittliche Genauigkeit von 87,43 % erreicht.
    - Nach zusätzlicher Feature Selection:
          - Mit SelectFromModel: durchschnittliche Genauigkeit 88,13 %
          - Mit RFECV: durchschnittliche Genauigkeit 90,05 %
Alle diese Ergebnisse liegen deutlich unter der 97 % Genauigkeit, die mit PS2 und 9 einfachen statistischen Merkmalen erzielt wurde.<br/>

- Zusammenfassung
    - PS2 mit wenigen, gut gewählten statistischen Merkmalen erweist sich als robuster, stabiler und leistungsfähiger als eine umfangreiche Feature-Extraktion mit tsfresh.
    - In diesem Projekt bringt komplexe Merkmalsextraktion keinen Vorteil gegenüber einfachen, domänennahen statistischen Features.





