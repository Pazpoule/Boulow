import numpy
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics as stat
from math import inf, sqrt
from scipy.optimize import minimize
import scipy.stats


#liste de 20 couleurs pour différencier les indices ou valeurs
list20Couleurs = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

nbrJourOuvre = 254
listETF = ['ISHS FTSE 250', 'HSBC FTSE 250', 'VAN FTSE 250', 'Amundi China', 'Lyxor China', 'Amundi Brazil', 'Amundi RUSSELL 2000', 'AMUNDI S&P 500', 'Amundi FTSE Italia', 'Lyxor FTSE 100', 'Lyxor Water', 'Lyxor Immo Euro', 'Lyxor ObligEtat Euro', 'Lyxor Japan', 'Lyxor Russia', 'Lyxor PEA PME (DR) UCITS ETF - Dist', 'Lyxor Korea', 'Lyxor India', 	'Lyxor CACx2', 	'Lyxor Euro 600 Tech', 	'Lyxor Euro 600 Goods', 	'Lyxor ESG', 	'Lyxor Euro 600 oil', 	'Lyxor DAX', 	'Lyxor Euro 600 Automobile', 	'Lyxor Euro 600 Chemicals', 	'Lyxor Greece', 	'Lyxor Euro 600 retail', 	'Nasdaq x2', 	'Amundi EM', 	'Lyxor EM', 	'Amundi Europe Small Cap', 	'Lyxor EMU Small Cap', 	'BNP Europe Small CAP', 	'iShares Core MSCI EMU', 	'Amundi NASDAQ 100', 	'Amundi USA', 	'Lyxor wd', 	'Lyxor DowJones', 	'Lyxos CAC MID60', 	'Amundi Japan', 	'SPDR Europe small cap', 	'Amundi Euro', 	'Amundi wd', 	'Lyxor PEA PME']
dataETF = pd.read_csv('C:/Users/C00000404/Documents/DossierPartage/Actif/BaseETF.csv')
listDate = list(dataETF['Date'])
dataETF.drop(['Date'], axis=1, inplace=True)

listPlus1800 = []
for indexColonne, colonne in enumerate(listETF):
    if len([dataETF.iloc[i, indexColonne] for i in range(len(dataETF)) if dataETF.iloc[i, indexColonne] != 0]) >= 1800:
        listPlus1800.append(colonne)
dataETFCut = dataETF[listPlus1800][:1800]

# On créer les dico par horizon
listHorizonETF = [annee*nbrJourOuvre for annee in range(1, 6)]
dicoRendementParHorizonETF = {}
dicoRendementParHorizonETFAnnualise = {}
for horizon in listHorizonETF:
    dicoRendementParHorizonETF[horizon] = []
    dicoRendementParHorizonETFAnnualise[horizon] = []
    for colonne in range(len(dataETFCut.columns)):
        dicoRendementParHorizonETF[horizon].append(np.array([(dataETFCut.iloc[ligne, colonne]-dataETFCut.iloc[ligne+horizon, colonne])/dataETFCut.iloc[ligne+horizon, colonne] for ligne in range(len(dataETFCut)-horizon)]))
        dicoRendementParHorizonETFAnnualise[horizon].append(np.array([((1 + (dataETFCut.iloc[ligne, colonne] - dataETFCut.iloc[ligne + horizon, colonne]) / dataETFCut.iloc[ligne + horizon, colonne])**(365/horizon)) -1 for ligne in range(len(dataETFCut) - horizon)]))

# On plot les densité
horizon = 5*nbrJourOuvre
figAnnualise = ff.create_distplot(dicoRendementParHorizonETFAnnualise[horizon], dataETFCut.columns, histnorm='probability', show_rug=False, colors=list20Couleurs, bin_size=.5)
figAnnualise.update_traces(marker={"opacity": 0.1})
figAnnualise.show()

# On créer les indicateurs Var et Cvar
alpha = 0.05
listQuantileETF = [np.quantile(dicoRendementParHorizonETFAnnualise[horizon][numETF], alpha) for numETF in range(len(dataETFCut.columns))]
listCVarETF = [stat.mean([dicoRendementParHorizonETFAnnualise[horizon][numIndice][rendement] for rendement in range(len(dicoRendementParHorizonETFAnnualise[horizon][numIndice])) if dicoRendementParHorizonETFAnnualise[horizon][numIndice][rendement]<=listQuantileETF[numIndice]]) for numIndice in range(len(dataETFCut.columns))]
dataQuantileETF = pd.DataFrame({'ETF': dataETFCut.columns, 'Var': listQuantileETF, 'CVar': listCVarETF}).sort_values(by='CVar', ascending=False, ignore_index=True)

# On créer une base de rendement et une base normalisé commencan par 100
dataETFCutRendement = dataETFCut.copy()
for indiceColonne, colonne in enumerate(dataETFCut.columns):
    for indiceLigne, ligne in enumerate(dataETFCut.index[:-1]):
        dataETFCutRendement.iloc[indiceLigne, indiceColonne] = (dataETFCut.iloc[indiceLigne, indiceColonne] - dataETFCut.iloc[indiceLigne + 1, indiceColonne]) / dataETFCut.iloc[indiceLigne + 1, indiceColonne]
    dataETFCutRendement.iloc[len(dataETFCutRendement) - 1, indiceColonne] = 100
dataETFCutNormalise = dataETFCutRendement.copy()
for indiceColonne, colonne in enumerate(dataETFCut.columns):
    for indiceLigne in range(len(dataETFCut.index) - 2, -1, -1):
        dataETFCutNormalise.iloc[indiceLigne, indiceColonne] = (dataETFCutRendement.iloc[indiceLigne, indiceColonne] + 1) * dataETFCutNormalise.iloc[indiceLigne + 1, indiceColonne]

data = dataETFCutNormalise










