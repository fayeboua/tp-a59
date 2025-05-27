# Importation des bibliothèques nécessaires
import os                             # Pour manipuler le système de fichiers
import yfinance as yf                 # Pour télécharger des données financières depuis Yahoo Finance
import pandas as pd                   # Pour manipuler les données tabulaires
import numpy as np                    # Pour la manipulation efficace de tableaux numériques
from sklearn.preprocessing import StandardScaler  # Pour normaliser les données (centrer-réduire)


#  Lecture de données locales (CSV)
def get_data(col='close'):
  """ Returns a 3 x n_step array """
  msft = pd.read_csv('data/daily_MSFT.csv', usecols=[col])
  ibm = pd.read_csv('data/daily_IBM.csv', usecols=[col])
  qcom = pd.read_csv('data/daily_QCOM.csv', usecols=[col])
  # Inverser les valeurs pour que les plus anciennes soient en premier
  #return np.array([msft[col].values[::-1],
  #                 ibm[col].values[::-1],
  #                 qcom[col].values[::-1]])
  # Exemple actuel limité à MSFT uniquement
  return np.array([msft[col].values[::-1]])

# Téléchargement via yfinance
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

    # Mise au format : renommage des colonnes pour plus de clarté et homogénéité
    datayf = datayf.reset_index()  # ramène la date comme colonne
    datayf = datayf.rename(columns={
    'Date': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
    })
    # Sélectionner uniquement les colonnes d'intérêt
    datayf = datayf[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # garde uniquement ces colonnes
    # Supprimer le nom du niveau de colonne (évite "Price" en haut des colonnes)
    datayf.columns.name = None
    #print("data Test Yahoo Finance:\n", datayf.head())
    return np.array([datayf[col].values])

# Création d’un scaler (Transformer et normaliser l’espace d’observation pour l’entraînement d’un agent )
def get_scaler(env):
  """ Prend un environnement de trading et retourne un scaler StandardScaler adapté à l'espace d'observation """

  # Définir le vecteur minimal des observations (tout à 0)
  low = [0] * (env.n_stock * 2 + 1)  # nombre d'actions possédées, prix d’actions, cash

  # Construire les bornes supérieures
  high = []
  max_price = env.stock_price_history.max(axis=1)  # prix max par stock
  min_price = env.stock_price_history.min(axis=1)  # prix min par stock

  max_cash = env.init_invest * 3  # valeur maximale supposée de cash (ex: 3x capital initial)
  max_stock_owned = max_cash // min_price  # estimation du max d’actions possédées par titre

  # Concaténation des composantes pour les bornes hautes
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  # Création du scaler à partir des bornes basse et haute
  scaler = StandardScaler()
  scaler.fit([low, high])  # standardisation basée sur ces deux extrêmes

  return scaler



def maybe_make_dir(directory):
  """ Crée un dossier s’il n’existe pas déjà """
  if not os.path.exists(directory):
    os.makedirs(directory)