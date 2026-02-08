# Heart Disease Prediction using Logistic Regression

This project implements a Logistic Regression model to predict the 10-year risk of Coronary Heart Disease (CHD) based on patient data from the Framingham Heart Study.

## Project Structure
- `logistic_reg_main.ipynb`: The original analysis and model development notebook.
- `model.py`: Script to clean data, train the model, and save the model and scaler as pickle files.
- `app.py`: Streamlit frontend application for making predictions.
- `requirements.txt`: List of dependencies.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train and Save Model:**
   ```bash
   python3 model.py
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Dataset
The dataset used is the Framingham Heart Study dataset, which includes demographic, behavioral, and medical risk factors for CHD.
