# Retail Sales Prediction Engine

This application deploys the **Stepwise Selection Regression Model** (Selected at Step 9, R² = 0.6655, Validation ASE = 617,048.48).

## Project Structure
- `app.py`: Streamlit application code.
- `requirements.txt`: Python package dependencies.
- `sample_batch_input.csv`: Sample CSV file for testing batch predictions.
- `.streamlit/config.toml`: Custom Streamlit theme configuration.

## Quick Start (Local)

1. Extract the zip archive and navigate into the folder:
   ```bash
   cd sales_prediction_app
   ```

2. Create and activate a virtual environment:
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the web application:
   ```bash
   streamlit run app.py
   ```

## Cloud Deployment (Streamlit Community Cloud)

1. Push all files to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub repository and set `app.py` as the main file path.
4. Click **Deploy**.
