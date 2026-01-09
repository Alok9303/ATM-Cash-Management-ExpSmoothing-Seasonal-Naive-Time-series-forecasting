# ATM Cash Forecasting System

A time series forecasting system for predicting ATM cash withdrawals and optimizing cash distribution across ATM networks using real-world data from the NN5 forecasting competition.

## 📊 Project Overview

This project analyzes historical ATM withdrawal data to:
- Predict future cash demand for individual ATMs
- Optimize cash distribution across an ATM network
- Minimize cash shortages while reducing idle cash

The system uses ensemble forecasting methods combining exponential smoothing and seasonal patterns to achieve robust predictions.

## 🎯 Key Features

- **Time Series Forecasting**: Implements multiple forecasting algorithms
  - Exponential Smoothing for trend capture
  - Seasonal Naive for weekly pattern recognition
  - Ensemble methods for improved accuracy

- **Missing Value Handling**: Intelligent imputation using day-of-week median values

- **Performance Metrics**: MAPE and SMAPE for forecast accuracy evaluation

- **Cash Optimization**: Priority-based algorithm for optimal cash distribution

- **Network Analysis**: Comprehensive ATM network statistics and insights

## 📁 Dataset

Uses the NN5 dataset from the Neural Forecasting Competition:
- **Source**: Real UK ATM withdrawal data
- **Format**: Daily cash withdrawal amounts
- **Coverage**: Multiple ATM locations over extended periods
- **File**: `nn5_daily_dataset_with_missing_values.txt`

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **datetime**: Time series handling

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/atm-cash-forecasting.git
cd atm-cash-forecasting

# Install required packages
pip install pandas numpy

# Ensure you have the dataset file
# Place nn5_daily_dataset_with_missing_values.txt in the project root
```

## 🚀 Usage

```bash
python test_nn5_system.py
```

### Output

The system generates:
1. **Network Analysis**: Statistics on ATM performance and volatility
2. **Forecast Evaluation**: Accuracy metrics (MAPE/SMAPE) on test data
3. **Cash Distribution Plan**: Priority-ranked allocation recommendations

## 📊 Sample Output

```
NN5 ATM NETWORK ANALYSIS
======================================================================
Total ATMs analyzed: 111
Total observations: 79,100
Date range: 1996-03-18 to 1998-06-30

Withdrawal Statistics:
  Network average daily withdrawal: $45,234.56
  
FORECAST ACCURACY EVALUATION
======================================================================
Average MAPE:  28.45%
Average SMAPE: 24.32%

CASH DISTRIBUTION OPTIMIZATION
======================================================================
Available cash: $500,000

Top 10 ATMs requiring urgent cash:
 1. T42: Days to empty:   0.8, Allocate: $45,000, Daily demand: $15,234.50
 2. T15: Days to empty:   1.2, Allocate: $38,500, Daily demand: $12,456.80
 ...
```

## 🔍 Key Functions

### Data Processing
- `parse_nn5_dataset()`: Parses NN5 format data
- `nn5_to_dataframe()`: Converts to pandas DataFrame
- `handle_missing_values_median()`: Imputes missing values

### Forecasting Methods
- `exponential_smoothing()`: Exponential smoothing forecast
- `seasonal_naive()`: Seasonal pattern-based forecast
- `ensemble_forecast()`: Combined forecast approach

### Evaluation
- `calculate_mape()`: Mean Absolute Percentage Error
- `calculate_smape()`: Symmetric MAPE
- `evaluate_forecasts()`: Test set evaluation

### Optimization
- `optimize_distribution()`: Cash allocation optimization
- `analyze_atm_network()`: Network statistics

## 📈 Methodology

### Forecasting Approach

1. **Exponential Smoothing** (α = 0.3)
   - Captures overall trends
   - Recent data weighted more heavily
   - Formula: `Prediction = α × New_Value + (1-α) × Old_Prediction`

2. **Seasonal Naive** (7-day cycle)
   - Leverages weekly patterns
   - Uses same-day-of-week from previous week
   - Effective for capturing periodic behavior

3. **Ensemble Method**
   - Averages both approaches
   - Reduces individual method weaknesses
   - More robust predictions

### Cash Distribution Logic

- Calculates **days-to-empty** for each ATM
- Prioritizes ATMs closest to running out
- Allocates 7 days worth of cash per ATM
- Maximizes network uptime with limited resources

## 📊 Performance Metrics

- **MAPE**: Measures average prediction error percentage
  - < 10%: Excellent
  - 10-20%: Good
  - 20-30%: Acceptable
  - > 30%: Needs improvement

- **SMAPE**: Symmetric version handling extreme values better

## 🔧 Customization

Adjust parameters in the code:

```python
# Forecasting parameters
alpha = 0.3              # Exponential smoothing factor
season_length = 7        # Weekly pattern
forecast_periods = 56    # 8 weeks ahead

# Distribution parameters
available_cash = 500000  # Total cash available
```

## 📚 Potential Improvements

- [ ] Add more sophisticated forecasting models (ARIMA, Prophet)
- [ ] Implement machine learning approaches
- [ ] Add visualization dashboard
- [ ] Real-time data integration
- [ ] Holiday effect modeling
- [ ] Multi-step ahead forecasting
- [ ] Confidence intervals for predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- NN5 Forecasting Competition for providing the dataset
- Neural forecasting research community

---

⭐ If you find this project helpful, please consider giving it a star!
