

A small project that prepares data and trains a model to predict whether a bank (or similar event) is open based on game and odds data. This repository contains raw and processed data, a lightweight data pipeline, and training code.

Repository layout
- data/raw/ - original data sources (CSV files used as input)
- data/processed/ - processed datasets produced by the pipeline (e.g. merged_features.csv)
- models/ - model-related artifacts and metadata (e.g. feature_names.txt)
- src/ - Python scripts for the data pipeline and training (entrypoints: data_pipeline.py, train_model.py)
- requirements.txt - Python dependencies

Quickstart

1. Clone the repository (see COMMANDS.md for different options):

powershell
git clone https://github.com/amruthxa/bank_is_open.git
cd bank_is_open


2. Create and activate a Python virtual environment (PowerShell):

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1


3. Install dependencies:

powershell
pip install --upgrade pip
pip install -r requirements.txt


4. Run the data pipeline to produce processed features:

powershell
python src/data_pipeline.py


5. Train the model:

powershell
python src/train_model.py


Useful files
- data/processed/merged_features.csv - merged feature table used for training
- models/feature_names.txt - list of model features used by training code

