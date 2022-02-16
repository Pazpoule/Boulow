# Simulation pour tester le modèle
import math
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

nbrJourOuvre = 254

dataCours = pd.DataFrame()
nbrSimu = 120*nbrJourOuvre
sd_base = 0.01
for nbrCours in range(20):
    print("Cours - ", nbrCours)
    bruit = np.random.normal(0, sd_base, nbrSimu)
    cours = [100]
    for i in range(nbrSimu):
        cours = [cours[0] * (1 + bruit[i])] + cours
    dataCours[str(nbrCours)] = cours

dataCoursCorrele = pd.DataFrame()
for nbrCours in range(len(dataCours.columns) - 1):
    print("Correlation - ", nbrCours)
    bruit = np.random.normal(0, sd_base, nbrSimu + 1)
    dataCoursCorrele[str(nbrCours)] = dataCours[dataCours.columns[nbrCours]] + dataCours[dataCours.columns[nbrCours + 1]] + bruit
    dataCoursCorrele[str(nbrCours)] = dataCoursCorrele[str(nbrCours)] * 100 / dataCoursCorrele.loc[len(dataCoursCorrele) - 1, str(nbrCours)]

for i in range(len(dataCoursCorrele.columns) - 1):
    print(np.corrcoef(dataCoursCorrele[dataCoursCorrele.columns[i]], dataCoursCorrele[dataCoursCorrele.columns[i + 1]])[0, 1])

for i in range(19):
    plt.plot(dataCoursCorrele[str(i)])

data = dataCoursCorrele

