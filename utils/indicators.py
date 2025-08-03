import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def calcRSI(data, P=14):
  try:
    # Calculate gains and losses
    data['diff_close'] = data['Close'] - data['Close'].shift(1)
    data['gain'] = np.where(data['diff_close']>0,
      data['diff_close'], 0)
    data['loss'] = np.where(data['diff_close']<0,
      np.abs(data['diff_close']), 0)
    
    # Get initial values
    data[['init_avg_gain', 'init_avg_loss']] = data[
      ['gain', 'loss']].rolling(P).mean()
    # Calculate smoothed avg gains and losses for all t > P
    avg_gain = np.zeros(len(data))
    avg_loss = np.zeros(len(data))
    
    for i, _row in enumerate(data.iterrows()):
      row = _row[1]
      if i < P - 1:
        last_row = row.copy()
        continue
      elif i == P-1:
        avg_gain[i] += row['init_avg_gain']
        avg_loss[i] += row['init_avg_loss']
      else:
        avg_gain[i] += ((P - 1) * avg_gain[i-1] + row['gain']) / P
        avg_loss[i] += ((P - 1) * avg_loss[i-1] + row['loss']) / P
      
      last_row = row.copy()
    
    data['avg_gain'] = avg_gain
    data['avg_loss'] = avg_loss
    # Calculate RS and RSI
    data['RS'] = data['avg_gain'] / data['avg_loss']
    data['RSI'] = 100 - 100 / (1 + data['RS'])    
    
  except Exception as e:
    print("Error message:", str(e))
    print(data)
  
  return data

def calcStochOscillator(data, N=14):
  try:
    data['low_N'] = data['RSI'].rolling(N).min()
    data['high_N'] = data['RSI'].rolling(N).max()
    data['StochRSI'] = 100 * (data['RSI'] - data['low_N']) / (data['high_N'] - data['low_N'])
  
  except Exception as e:
      print("Error message:", str(e))
      print(data)
  
  return data

def calcStochRSI(data, P=14, N=14):
  data = calcRSI(data, P)
  data = calcStochOscillator(data, N)
  return data

def calcMACD(data, P=12, Q=26, R=9):
  try:
    data['EMA_P'] = data['Close'].ewm(span=P).mean()
    data['EMA_Q'] = data['Close'].ewm(span=Q).mean()
    data['MACD'] = data['EMA_P'] - data['EMA_Q']
    data['MACD_signal'] = data['MACD'].ewm(span=R).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
      
  except Exception as e:
    print("Error message:", str(e))
    print(data)
    
  return data

def calcReturns(df):
  # Helper function to avoid repeating too much code
  df['returns'] = df['Close'] / df['Close'].shift(1)
  df['log_returns'] = np.log(df['returns'])
  df['strat_returns'] = df['position'].shift(1) * df['returns']
  df['strat_log_returns'] = df['position'].shift(1) * df['log_returns']
  df['cum_returns'] = np.exp(df['log_returns'].cumsum()) - 1
  df['strat_cum_returns'] = np.exp(df['strat_log_returns'].cumsum()) / - 1
  df['peak'] = df['cum_returns'].cummax()
  df['strat_peak'] = df['strat_cum_returns'].cummax()
  return df
