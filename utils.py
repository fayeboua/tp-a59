import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(col='close'):
  """ Returns a 3 x n_step array """
  msft = pd.read_csv('data/daily_MSFT.csv', usecols=[col])
  ibm = pd.read_csv('data/daily_IBM.csv', usecols=[col])
  qcom = pd.read_csv('data/daily_QCOM.csv', usecols=[col])
  # recent price are at top; reverse it
  #return np.array([msft[col].values[::-1],
  #                 ibm[col].values[::-1],
  #                 qcom[col].values[::-1]])
  return np.array([msft[col].values[::-1]])

def get_data_yf(col='close'):
    # Téléchargement des données
    #symbol = "AAPL" #Apple Inc.
    symbol = "MSFT" #Microsoft Corp.
    # symbol = "GOOGL" #Alphabet Inc.
    # symbol = "AMZN" #Amazon.com Inc.
    #symbol = "TSLA" #Tesla Inc.
    datayf = yf.download(symbol, start="2000-01-01", end="2025-12-31", group_by='ticker')
    if isinstance(datayf.columns, pd.MultiIndex):
         datayf = datayf[symbol]  # garde uniquement les colonnes pour le ticker AAPL
    #close_prices = datayf["Close"].values.flatten()
    #print("Columns:", datayf.columns)

    datayf = datayf.reset_index()  # ramène la date comme colonne
    datayf = datayf.rename(columns={
    'Date': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
    })
    datayf = datayf[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # garde uniquement ces colonnes
    # Supprimer le nom du niveau de colonne (évite "Price" en haut des colonnes)
    datayf.columns.name = None
    #print("data Test Yahoo Finance:\n", datayf.head())
    return np.array([datayf[col].values])

def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3 # 3 is a magic number...
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)