import pandas as pd
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
import optuna

import Actif.TraitementETF
# import Actif.simu_Cours_correle


def CoursPondere(w, data):
    '''retourne le cours du portefeuille sachant les poids'''
    return list(sum([data[colonne] * w[indiceColonne] for indiceColonne, colonne in enumerate(data.columns)]))

def plotCours(cours, data):
    '''On plot EN WEB un cours que l'on compare au cours des données'''
    figCours = px.line(data, x=data.index, y=list(cours))
    for numActif, actif in enumerate(data.columns):
        figCours.add_trace(go.Scatter(x=list(data.index), y=list(data[actif]), mode='lines', name=actif, line=go.scatter.Line(color=list20Couleurs[numActif])))
    figCours.show()

def coursTORend(cours, horizon:int):
    return [(1 + (cours[ligne] - cours[ligne + horizon]) / cours[ligne + horizon]) - 1 for ligne in range(len(cours) - horizon)]

def coursTORendAnnualise(cours, horizon:int):
    return [((1 + (cours[ligne] - cours[ligne + horizon]) / cours[ligne + horizon]) ** (nbrJourOuvre / horizon)) - 1 for ligne in range(len(cours) - horizon)]

def Strategy(w: list, data, horizon:int, alpha=0.05):
    '''On definit la FONCTION DE COUT !! - calcul -CVAR'''
    cours = CoursPondere(w, data)
    rendements = coursTORendAnnualise(cours, horizon)
    var = np.quantile(rendements, alpha)
    cVar = stat.mean([rendement for rendement in rendements if rendement <= var])
    return cVar

def fonctionDeCout(trial, data, horizon:int):
    ''' fonction de cout optuna
        definie les bornes pour chaque poids
        retourne -CVaR + contrainte '''
    w = [trial.suggest_float(f'{poids}', 0, 1) for poids in range(len(data.columns))]
    return -Strategy(w, data, horizon, alpha=alpha) #+ 100*(w[0]/sum(w)<0.05)

def dataTORendAnnualise(data, horizon):
    '''transforme les donnée en rendement annualise pour plot les densités'''
    RendementAnnualise = []
    for colonne in range(len(data.columns)):
        RendementAnnualise.append(np.array([((1 + (data.iloc[ligne, colonne] - data.iloc[ligne + horizon, colonne]) / data.iloc[ligne + horizon, colonne]) ** (nbrJourOuvre / horizon)) - 1 for ligne in range(len(data) - horizon)]))
    return RendementAnnualise



if __name__ == '__main__':

    # Paramètres
    list20Couleurs = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']
    nbrJourOuvre = 254
    horizon = 10 * nbrJourOuvre
    alpha = 0.05
    nombreEssaiesOptimisation = 10000
    data = Actif.simu_Cours_correle.data  # la base data doit etre d'un certain format, plusieurs cours, en temps reversed et normalisé commancant a 100
    corr = np.corrcoef([data[data.columns[i]] for i in range(len(data.columns))])  # On definit la matrice de correlation
    nbrSensibilite = 5000
    ecartSensibilite = 0.05

    # Strategie équipondéré
    w_equipondere = [1 / len(data.columns) for _ in range(len(data.columns))]
    cours_equipondere = CoursPondere(w_equipondere, data)
    plotCours(cours_equipondere, data)

    # Strategie optimale
    study = optuna.create_study()
    study.optimize(lambda x: fonctionDeCout(x, data, horizon), n_trials=nombreEssaiesOptimisation)
    # w_optimal = list(study.best_params.values())
    w_optimal = list(np.array(list(study.best_params.values()))/sum(list(study.best_params.values())))
    cours_optimal = CoursPondere(w_optimal, data)
    plotCours(cours_equipondere, data)

    # On affiche les Resultats
    print(f'Les poids optimaux sont :')
    print(w_optimal)
    print(f"On passe d'une cVar equipondérée de {round(Strategy(w_equipondere, data, horizon, alpha) * 100, 2)}% à une cVar solution de {round(Strategy(w_optimal, data, horizon, alpha) * 100, 2)}%")
    print(f'Rendement moyen annuel Equipondéré : {round(np.array(coursTORendAnnualise(cours_equipondere, horizon)).mean() * 100, 2)} %')
    print(f'Rendement moyen annuel Optimal : {round(np.array(coursTORendAnnualise(cours_optimal, horizon)).mean() * 100, 2)} %')
    plt.figure(figsize=[16, 9])
    plt.title("Comparaison de la strategie equipondérée à la strategie optimale")
    plt.plot(cours_equipondere)
    plt.plot(cours_optimal)

    # On plot les densité
    array_horizon = dataTORendAnnualise(data, horizon)
    array_solution = array_horizon + [np.array(coursTORendAnnualise(cours_equipondere, horizon)), np.array(coursTORendAnnualise(cours_optimal, horizon))]
    label_solution = list(data.columns)
    label_solution.append('Stratégie Equipondéré')
    label_solution.append('Stratégie Optimale')
    figAnnualise = ff.create_distplot(array_solution, label_solution, histnorm='probability', show_rug=False, colors=list20Couleurs, bin_size=.5)
    figAnnualise.update_traces(marker={"opacity": 0.1})
    figAnnualise.show()

    # On récupère les solutions
    dataSolution = pd.DataFrame({})
    dataSolution["Actifs"] = data.columns
    dataSolution["Poids"] = w_optimal

    # On fait des tests de sensibilité
    dataSensibilite = pd.DataFrame({str(column) : [w_optimal[indiceColumn]] for indiceColumn, column in enumerate(data.columns)})
    for _ in range(nbrSensibilite):
        dataSensibilite = dataSensibilite.append(pd.DataFrame({str(column) : [max(min(w_optimal[indiceColumn]+random.uniform(-ecartSensibilite, ecartSensibilite), 1), 0)] for indiceColumn, column in enumerate(data.columns)}), ignore_index=True)
    dataSensibilite["CVaR"] = 0
    for index in dataSensibilite.index:
        poidsTotal = sum([dataSensibilite.loc[index, column] for column in dataSensibilite.columns])
        for column in dataSensibilite.columns:
            dataSensibilite.loc[index, column] = dataSensibilite.loc[index, column] / poidsTotal
        dataSensibilite.loc[index, "CVaR"] = Strategy([dataSensibilite.loc[index, column] for column in dataSensibilite.columns], data, horizon, alpha)
    dataSensibilite.sort_values(by="CVaR", ignore_index=True, inplace=True)
    print(f'La sensibilité CVaR moyenne pour une sd de {ecartSensibilite * 100}% est de {round(np.mean(dataSensibilite["CVaR"]) * 100, 2)}%')








    # # On optimise par Scipy------------------------------------------------------------------------------------------------
    # horizon = 10*nbrJourOuvre
    # function = lambda w: Strategy(w, data, horizon, 0.2)
    # bnds = [(0, 1) for _ in w_equipondere]
    # def contrainte(w):
    #     return abs(sum(w) - 1) + sum([1 for x in w if x > 0.3])
    # cons = ({'type': 'eq', 'fun': contrainte})
    # w_solution = w_equipondere
    # print(Strategy(w_equipondere), contrainte(w_equipondere))
    #
    # for conditionInitial in range(4):
    #     print(conditionInitial)
    #     w_random = [random.uniform(0, 1) for _ in range(len(data.columns))]
    #     w_random = [x / sum(w_random) for x in w_random]
    #     essaie = minimize(lambda w: -function(w), x0=w_random, method='SLSQP', bounds=bnds, constraints=cons)
    #     essaie = essaie.x
    #     print(Strategy(essaie), contrainte(essaie))
    #     if Strategy(essaie) > Strategy(w_solution) and contrainte(essaie) < 0.1:
    #         w_solution = essaie
    # cours_solution = CoursPondere(w_solution, data)
    # # On plot le cours resultant
    # plotCours(cours_solution, data)
    # rend_solution = coursTORendAnnualise(cours_solution, horizon)
    # cours_equipondere = CoursPondere(w_equipondere, data)
    # rend_equipondere = coursTORendAnnualise(cours_equipondere, horizon)


    # # On optimise par grille -----------------------------------------------------------------------------------------------------
    # datatest = data[data.columns[:5]]
    # def CreateGrid(data, listContraintes, pas=20):
    #     tailleStepGrille = pas
    #     dataResultats = pd.DataFrame({f"Poids {column}": [0 for _ in range(int((100/tailleStepGrille)**len(data.columns)))] for column in data.columns})
    #     for index in range(1, int((100/tailleStepGrille)**len(data.columns))):
    #         dataResultats.loc[index, f"Poids {0}"] = (dataResultats.loc[index-1, f"Poids {0}"] + tailleStepGrille) % 100
    #         for indexColumn in range(1, len(dataResultats.columns[1:, ])+1):
    #             dataResultats.loc[index, f"Poids {indexColumn}"] = ((dataResultats.loc[index-1, f"Poids {indexColumn}"] + tailleStepGrille) % 100) if (dataResultats.loc[index-1, f"Poids {indexColumn-1}"]==100-tailleStepGrille) and (dataResultats.loc[index, f"Poids {indexColumn-1}"]==0) else dataResultats.loc[index-1, f"Poids {indexColumn}"]
    #     dataPondere = dataResultats.copy()
    #     for index in dataResultats.index:
    #         dataPondere.loc[index] = dataResultats.loc[index]/sum(dataResultats.loc[index])
    #     for indiceContrainte, contrainte in enumerate(listContraintes):
    #         dataPondere = dataPondere[(dataPondere[f"Poids {indiceContrainte}"]>=contrainte[0]) & (dataPondere[f"Poids {indiceContrainte}"]<contrainte[1])]
    #         dataPondere.reset_index(inplace=True, drop=True)
    #     return dataResultats, dataPondere
    #
    # dataGrid, dataGridPondere = CreateGrid(datatest, [[0.1, 1] for _ in range(len(datatest.columns))], 20)
    # for index in dataGridPondere.index:
    #     dataGridPondere.loc[index, "CVaR"] = Strategy(list(dataGridPondere.loc[index]), data = datatest, horizon = 10*nbrJourOuvre, alpha = 0.05)
    # dataGridPondere.sort_values(by="CVaR", inplace=True, ignore_index=True)
    #
    # [0.5, 0.15], [0.5, 0.15], [0.35, 0.45], [0.5, 0.15], [0.25, 0.35]
    # output = CoursPondere(list(dataGridPondere.loc[len(dataGridPondere)-1])[:-1], datatest)

