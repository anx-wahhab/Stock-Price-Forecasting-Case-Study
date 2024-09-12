# Stock Price Prediction Case Study: Amazon and J&J

This repository contains the code and analysis for predicting stock prices using multiple time series forecasting models. The analysis focuses on Amazon (AMZN) and Johnson & Johnson (J&J) stock data over a specified period, employing a combination of traditional statistical models and deep learning techniques.

## Dataset
- **Amazon (AMZN)**: Data includes the closing price for Amazon’s stock. A full preprocessing pipeline was applied, including data cleaning, stationarity checks, and differentiation.
- **Johnson & Johnson (J&J)**: Similar preprocessing steps were applied to J&J's stock data.
  
Both datasets were preprocessed for time series analysis, with stationarity checks performed using Augmented Dickey-Fuller (ADF) tests. Differencing and Fourier transformations were applied to make the data suitable for ARIMA and deep learning models.

## Project Structure
The project explores the following models:
1. **ARIMA (AutoRegressive Integrated Moving Average)**:
   - Used for univariate time series forecasting after transforming the data to be stationary.
   - Fourier transformation was integrated to enhance ARIMA's ability to capture periodic patterns.
2. **LSTM (Long Short-Term Memory)**:
   - A deep learning model that captures long-term dependencies in sequential data.
3. **GRU (Gated Recurrent Unit)**:
   - A variant of LSTM with a simpler architecture, used to compare performance against LSTM.

### Steps in the Analysis:
1. **Data Preprocessing**: Stationarity checks and differencing.
2. **Modeling**:
   - ARIMA, Fourier-ARIMA, LSTM, and GRU models were trained on both datasets.
3. **Evaluation**: Root Mean Square Error (RMSE) was used to evaluate model performance.

## Key Results
### Amazon Stock Forecasting
- **ARIMA** (Baseline): RMSE = 3.05
- **Fourier-Transformed ARIMA**: RMSE = 0.075
- **LSTM**: RMSE = 0.55
- **GRU**: RMSE = 0.95

### J&J Stock Forecasting
- **ARIMA** (Baseline): RMSE = 4.02
- **Fourier-Transformed ARIMA**: RMSE = 0.19
- **LSTM**: RMSE = 1.69
- **GRU**: RMSE = 1.69

Fourier-Transformed ARIMA showed the best performance in both cases.

## Tools & Libraries
- **Python**: For data analysis and model implementation.
- **Libraries**:
  - `pandas`, `matplotlib`, `statsmodels`, `scikit-learn`
  - `torch` for LSTM and GRU models
  - Fourier transformation via `scipy.fft`
  
## How to Use
1. Clone the repository.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the provided Jupyter notebook to replicate the analysis.

## Report
The detailed analysis report can be found [here](./Report.pdf) covering the methodology, evaluation, and results in depth.
