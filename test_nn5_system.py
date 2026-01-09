"""
Test the NN5 forecasting system with real data
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# PARSE NN5 DATA
# ============================================================================

def parse_nn5_dataset(data_string):
    """
    Parse the NN5 dataset format
    Format: T1:1996-03-18 00-00-00:value1,value2,value3,...
    """
    
    lines = [line.strip() for line in data_string.strip().split('\n') 
             if line.strip() and not line.startswith('#') and not line.startswith('@')]
    
    dataset = {}
    
    for line in lines:
        if ':' not in line:
            continue
        
        try:
            parts = line.split(':')
            series_id = parts[0]
            start_date_str = parts[1]
            values_str = parts[2]
            
            # Parse start date
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H-%M-%S")
            
            # Parse values - handle missing values (0 or empty)
            values = []
            for v in values_str.split(','):
                try:
                    val = float(v)
                    values.append(val if val != 0 else np.nan)
                except:
                    values.append(np.nan)
            
            # Create date range
            dates = [start_date + timedelta(days=i) for i in range(len(values))]
            
            dataset[series_id] = {
                'dates': dates,
                'values': values,
                'start_date': start_date,
                'n_observations': len(values)
            }
        except Exception as e:
            print(f"Error parsing line: {e}")
            continue
    
    return dataset


def nn5_to_dataframe(dataset):
    """Convert NN5 dataset to pandas DataFrame"""
    data = []
    
    for series_id, series_data in dataset.items():
        for date, value in zip(series_data['dates'], series_data['values']):
            data.append({
                'date': date,
                'atm_id': series_id,
                'withdrawal': value,
                'day_of_week': date.weekday(),
                'day_of_month': date.day,
                'month': date.month,
                'year': date.year,
                'is_weekend': 1 if date.weekday() >= 5 else 0,
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['atm_id', 'date']).reset_index(drop=True)
    
    return df


# ============================================================================
# SIMPLE FORECASTING
# ============================================================================

def handle_missing_values_median(df):
    """Replace missing values with median of same day-of-week"""
    
    for atm_id in df['atm_id'].unique():
        for day_of_week in range(7):
            mask = (df['atm_id'] == atm_id) & (df['day_of_week'] == day_of_week)
            
            if mask.sum() > 0:
                median_val = df[mask]['withdrawal'].median()
                
                # Replace NaN and 0 with median
                nan_mask = mask & (df['withdrawal'].isna())
                zero_mask = mask & (df['withdrawal'] == 0)
                
                df.loc[nan_mask, 'withdrawal'] = median_val
                df.loc[zero_mask, 'withdrawal'] = median_val
    
    return df


def exponential_smoothing(series, alpha=0.3, forecast_periods=56):
    """Exponential smoothing forecast"""
    series = np.array(series)
    series = series[~np.isnan(series)]
    
    if len(series) < 2:
        return np.array([series[0]] * forecast_periods) if len(series) > 0 else np.array([0] * forecast_periods)
    
    result = [series[0]]
    for i in range(1, len(series)):
        result.append(alpha * series[i] + (1 - alpha) * result[i-1])
    
    forecast = [result[-1]] * forecast_periods
    return np.array(forecast)


def seasonal_naive(series, season_length=7, forecast_periods=56):
    """Use same day from previous week"""
    series = np.array(series)
    series = series[~np.isnan(series)]
    
    if len(series) < season_length:
        return np.array([np.mean(series)] * forecast_periods) if len(series) > 0 else np.array([0] * forecast_periods)
    
    forecast = []
    for i in range(forecast_periods):
        idx = len(series) - season_length - 1
        if idx >= 0:
            forecast.append(series[idx])
        else:
            forecast.append(np.mean(series))
    
    return np.array(forecast)


def ensemble_forecast(series, forecast_periods=56):
    """Ensemble of multiple simple models"""
    exp = exponential_smoothing(series, alpha=0.3, forecast_periods=forecast_periods)
    seasonal = seasonal_naive(series, season_length=7, forecast_periods=forecast_periods)
    
    # Equal weight
    ensemble = (exp + seasonal) / 2
    return ensemble


# ============================================================================
# CALCULATE METRICS
# ============================================================================

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    mask = actual != 0
    if not mask.any():
        return np.inf
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_smape(actual, predicted):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = np.abs(actual) + np.abs(predicted)
    mask = denominator != 0
    
    if not mask.any():
        return np.inf
    
    return 100 * np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denominator[mask])


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_atm_network(df):
    """Analyze ATM network statistics"""
    
    stats = defaultdict(dict)
    
    for atm_id in df['atm_id'].unique():
        atm_data = df[df['atm_id'] == atm_id]['withdrawal']
        atm_data = atm_data[~atm_data.isna()]
        
        if len(atm_data) > 0:
            stats[atm_id] = {
                'mean': atm_data.mean(),
                'std': atm_data.std(),
                'min': atm_data.min(),
                'max': atm_data.max(),
                'cv': atm_data.std() / atm_data.mean() if atm_data.mean() > 0 else 0
            }
    
    return stats


def print_network_summary(df, stats):
    """Print summary of network"""
    
    print("\n" + "="*70)
    print("NN5 ATM NETWORK ANALYSIS")
    print("="*70)
    
    means = [s['mean'] for s in stats.values()]
    cvs = [s['cv'] for s in stats.values()]
    
    print(f"Total ATMs analyzed: {len(stats)}")
    print(f"Total observations: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nWithdrawal Statistics:")
    print(f"  Network average daily withdrawal: ${np.mean(means):.2f}")
    print(f"  Min ATM: ${np.min(means):.2f}")
    print(f"  Max ATM: ${np.max(means):.2f}")
    print(f"\nVolatility (Coefficient of Variation):")
    print(f"  Average CV: {np.mean(cvs):.3f}")
    print(f"  Most stable (min CV): {np.min(cvs):.3f}")
    print(f"  Most volatile (max CV): {np.max(cvs):.3f}")


# ============================================================================
# FORECASTING EVALUATION
# ============================================================================

def evaluate_forecasts(df, test_size=56):
    """Evaluate forecast accuracy on holdout set"""
    
    print("\n" + "="*70)
    print("FORECAST ACCURACY EVALUATION")
    print("="*70)
    print(f"Test set: Last {test_size} days\n")
    
    metrics = defaultdict(dict)
    all_mape = []
    all_smape = []
    
    for atm_id in df['atm_id'].unique():
        atm_data = df[df['atm_id'] == atm_id].sort_values('date')
        
        n = len(atm_data)
        if n <= test_size + 30:  # Need at least 30 days to train
            continue
        
        # Split
        train = atm_data.iloc[:-test_size]
        test = atm_data.iloc[-test_size:]
        
        # Forecast
        forecast = ensemble_forecast(train['withdrawal'].values, forecast_periods=test_size)
        actual = test['withdrawal'].values
        
        # Metrics
        mape = calculate_mape(actual, forecast)
        smape = calculate_smape(actual, forecast)
        mae = np.mean(np.abs(actual - forecast))
        
        metrics[atm_id] = {
            'mape': mape,
            'smape': smape,
            'mae': mae
        }
        
        if mape != np.inf:
            all_mape.append(mape)
        if smape != np.inf:
            all_smape.append(smape)
    
    # Summary
    if all_mape:
        avg_mape = np.mean(all_mape)
        avg_smape = np.mean(all_smape)
        
        print(f"Average MAPE:  {avg_mape:.2f}%")
        print(f"Average SMAPE: {avg_smape:.2f}%")
        print(f"Evaluated {len(metrics)} ATMs")
        
        # Best and worst
        sorted_mape = sorted(metrics.items(), key=lambda x: x[1]['mape'] if x[1]['mape'] != np.inf else 999)
        
        print(f"\nBest performing ATMs:")
        for atm, m in sorted_mape[:3]:
            print(f"  {atm}: MAPE {m['mape']:.2f}%")
        
        print(f"\nMost challenging ATMs:")
        for atm, m in sorted_mape[-3:]:
            print(f"  {atm}: MAPE {m['mape']:.2f}%")


# ============================================================================
# DISTRIBUTION OPTIMIZATION
# ============================================================================

def optimize_distribution(df, forecast_periods=56, available_cash=500000):
    """Optimize cash distribution"""
    
    print("\n" + "="*70)
    print("CASH DISTRIBUTION OPTIMIZATION")
    print("="*70)
    print(f"Available cash: ${available_cash:,.0f}\n")
    
    # Calculate average daily demand for each ATM
    forecasts = {}
    for atm_id in df['atm_id'].unique():
        atm_data = df[df['atm_id'] == atm_id]['withdrawal']
        atm_data = atm_data[~atm_data.isna()]
        
        if len(atm_data) > 0:
            daily_avg = atm_data.mean()
            forecasts[atm_id] = {
                'daily_avg': daily_avg,
                'weekly_demand': daily_avg * 7,
                'forecast': ensemble_forecast(atm_data.values, forecast_periods=forecast_periods)
            }
    
    # Calculate priority (days to empty)
    priorities = {}
    for atm_id, forecast_data in forecasts.items():
        # Simulate current balance (3 days worth)
        current_balance = forecast_data['daily_avg'] * 3
        days_to_empty = current_balance / forecast_data['daily_avg'] if forecast_data['daily_avg'] > 0 else np.inf
        
        priority = 1 / (days_to_empty + 0.5)
        priorities[atm_id] = {
            'days_to_empty': days_to_empty,
            'daily_avg': forecast_data['daily_avg'],
            'priority': priority
        }
    
    # Sort by priority and allocate
    sorted_atms = sorted(priorities.items(), key=lambda x: x[1]['priority'], reverse=True)
    
    allocation = {}
    remaining = available_cash
    
    for atm_id, priority_data in sorted_atms:
        target = priority_data['daily_avg'] * 7  # 7 days
        amount = min(target, remaining)
        allocation[atm_id] = amount
        remaining -= amount
    
    # Display top priorities
    print("Top 10 ATMs requiring urgent cash:\n")
    for i, (atm_id, priority_data) in enumerate(sorted_atms[:10], 1):
        print(f"{i:2d}. {atm_id}: Days to empty: {priority_data['days_to_empty']:5.1f}, "
              f"Allocate: ${allocation[atm_id]:10,.0f}, Daily demand: ${priority_data['daily_avg']:8.2f}")
    
    total_allocated = sum(allocation.values())
    unallocated = available_cash - total_allocated
    
    print(f"\nTotal allocated: ${total_allocated:,.0f}")
    print(f"Unallocated: ${unallocated:,.0f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis"""
    
    print("="*70)
    print("NN5 ATM CASH FORECASTING SYSTEM")
    print("Real UK ATM Withdrawal Data (Neural Forecasting Competition)")
    print("="*70)
    
    # Load data
    print("\n[1] Loading NN5 dataset...")
    with open('nn5_daily_dataset_with_missing_values.txt', 'r') as f:
        data_string = f.read()
    
    dataset = parse_nn5_dataset(data_string)
    print(f"    Loaded {len(dataset)} ATM series")
    
    # Convert to DataFrame
    print("\n[2] Processing data...")
    df = nn5_to_dataframe(dataset)
    print(f"    Shape: {df.shape}")
    print(f"    ATMs: {df['atm_id'].nunique()}")
    print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Handle missing values
    print("\n[3] Handling missing values...")
    df = handle_missing_values_median(df)
    missing_after = df['withdrawal'].isna().sum()
    print(f"    Remaining missing: {missing_after}")
    
    # Analyze network
    print("\n[4] Analyzing ATM network...")
    stats = analyze_atm_network(df)
    print_network_summary(df, stats)
    
    # Evaluate forecasts
    print("\n[5] Evaluating forecasts...")
    evaluate_forecasts(df, test_size=56)
    
    # Optimize distribution
    print("\n[6] Optimizing distribution...")
    optimize_distribution(df, available_cash=500000)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return df, stats


if __name__ == "__main__":
    df, stats = main()
