# Store Sales Prediction Streamlit App

This repository contains the full Streamlit web application for predicting store sales revenue and high-sales classification using machine learning models trained on transaction and economic data.

## Project Structure
- `app.py`: Streamlit dashboard web application.
- `train_models.py`: Python script to train models and serialize `model_pipeline.joblib`.
- `requirements.txt`: Python package dependencies.
- `GROUP 7 [raw data file].csv`: Dataset containing store features and sales targets.

## Setup Instructions

1. **Unzip the project files** into your workspace.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the models**:
   ```bash
   python train_models.py
   ```
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
