# AI-Driven Climate Temperature Prediction System

## Project Goal
Develop a Python-based AI system to predict future global temperature anomalies using historical temperature data and visualize results via a SCADA-style dashboard.

## Dataset
Berkeley Earth Climate Change — Global Land & Ocean Temperature Data  
[Dataset Link](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data/data)

## Features
- **Input:** Year / Time Index  
- **Output:** Predicted Temperature Anomaly

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt  
``` 

## Folder Structure
```  
ClimateChangeAI/
│
├── data/                         # Dataset and raw data files
│   └── raw_data.csv
│
├── gui/                          # Graphical User Interface (SCADA system)
│   ├── main_window.py
│   ├── plc_simulation_window.py
│   ├── process_window1.py
│   ├── process_window2.py
│   ├── statistics_window.py
│   └── __pycache__/              # Compiled Python cache files
│
├── model/                        # Model architecture and trained model
│   ├── climate_model.py
│   ├── trained_model.pth
│   └── __pycache__/
│
├── notebooks/                    # Jupyter notebooks for experimentation
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── utils/                        # Helper modules
│   ├── data_loader.py
│   ├── plot_functions.py
│   ├── predictions.py
│   └── __pycache__/
│
├── venv/                         # Virtual environment files
│   ├── bin/
│   ├── include/
│   ├── lib/
│   ├── share/
│   └── pyvenv.cfg
│
├── main.py                       # Main entry point for running the project
├── train_model.py                # Script for training the climate model
├── evaluate_model.py             # Script for evaluating the trained model
│
├── presentation.pptx             # Project presentation slides
├── report.docx                   # Final project report
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore rules
└── .python-version               # Python environment version
```
