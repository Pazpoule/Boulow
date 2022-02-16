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

# import Actif.TraitementETF
import Actif.simu_Cours_correle

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------Optimisation------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------
    list20Couleurs = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff',
                      '#000000']
    # Nombre de jour par an
    nbrJourOuvre = 254
    # la base data doit etre d'un certain faormat, plusieurs cours, en temps reversed et normalisé commancant a 100
    data = Actif.simu_Cours_correle.data

    # On prépare les éléments necessaire a la descente de gradiant
    w_equipondere = [1 / len(data.columns) for _ in range(len(data.columns))]

    # On definit le cours du portefeuille en fonction de la pondération
    def CoursPondere(w, data):
        return sum([data[colonne] * w[indiceColonne] for indiceColonne, colonne in enumerate(data.columns)])
    cours_equipondere = CoursPondere(w_equipondere, data)

    # On plot EN WEB un cours que l'on compare au cours des données
    def plotCours(cours, data):
        figCours = px.line(data, x=data.index, y=list(cours))
        for numActif, actif in enumerate(data.columns):
            figCours.add_trace(go.Scatter(x=list(data.index), y=list(data[actif]), mode='lines', name=actif, line=go.scatter.Line(color=list20Couleurs[numActif])))
        figCours.show()
    plotCours(cours_equipondere, data)

    # fonction de conversion du cours en rendement
    def coursTORend(cours, horizon):
        return [(1 + (cours[ligne] - cours[ligne + horizon]) / cours[ligne + horizon]) - 1 for ligne in range(len(cours) - horizon)]
    def coursTORendAnnualise(cours, horizon):
        return [((1 + (cours[ligne] - cours[ligne + horizon]) / cours[ligne + horizon]) ** (nbrJourOuvre / horizon)) - 1 for ligne in range(len(cours) - horizon)]

    # On definit la FONCTION DE COUT !! - calcul -CVAR
    def Strategy(w: list, data=data, horizon=nbrJourOuvre, alpha=0.05):
        cours = CoursPondere(w, data)
        rendements = coursTORendAnnualise(cours, horizon)
        var = np.quantile(rendements, alpha)
        cVar = stat.mean([rendement for rendement in rendements if rendement <= var])
        return cVar

    # On optimise ------------------------------------------------------------------------------------------------
    horizon = 10*nbrJourOuvre
    function = lambda w: Strategy(w, data, horizon, 0.2)
    bnds = [(0, 1) for _ in w_equipondere]
    def contrainte(w):
        return abs(sum(w) - 1) + sum([1 for x in w if x > 0.3])
    cons = ({'type': 'eq', 'fun': contrainte})
    w_solution = w_equipondere
    print(Strategy(w_equipondere), contrainte(w_equipondere))
    test = minimize(lambda w: -function(w), x0=w_equipondere, method='SLSQP', bounds=bnds, constraints=cons)
    test = test.x
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='CG', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='BFGS', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='Newton-CG', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='L-BFGS-B', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='TNC', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='COBYLA', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='SLSQP', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='dogleg', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='trust-constr', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='trust-ncg', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='trust-exact', bounds=bnds, constraints=cons).x))
    print(Strategy(minimize(lambda w: -function(w), x0=w_equipondere, method='trust-krylov', bounds=bnds, constraints=cons).x))




    print(Strategy(test.jac), contrainte(test))
    for conditionInitial in range(4):
        print(conditionInitial)
        w_random = [random.uniform(0, 1) for _ in range(len(data.columns))]
        w_random = [x / sum(w_random) for x in w_random]
        essaie = minimize(lambda w: -function(w), x0=w_random, method='SLSQP', bounds=bnds, constraints=cons)
        essaie = essaie.x
        print(Strategy(essaie), contrainte(essaie))
        if Strategy(essaie) > Strategy(w_solution) and contrainte(essaie) < 0.1:
            w_solution = essaie
    cours_solution = CoursPondere(w_solution, data)
    # On plot le cours resultant
    plotCours(cours_solution, data)
    rend_solution = coursTORendAnnualise(cours_solution, horizon)
    cours_equipondere = CoursPondere(w_equipondere, data)
    rend_equipondere = coursTORendAnnualise(cours_equipondere, horizon)

    print(f"On passe d'une cVar equipondérée de {-Strategy(w_equipondere)} à une cVar solution de {-Strategy(w_solution)}")

    # On plot les densité
    # On créer les dico par horizon, pour plusieurs horizon
    def dataTOArrayRend(data, horizon):
        RendementAnnualise = []
        for colonne in range(len(data.columns)):
            RendementAnnualise.append(np.array([((1 + (data.iloc[ligne, colonne] - data.iloc[ligne + horizon, colonne]) / data.iloc[ligne + horizon, colonne]) ** (nbrJourOuvre / horizon)) - 1 for ligne in range(len(data) - horizon)]))
        return RendementAnnualise
    array_horizon = dataTOArrayRend(data, horizon)
    array_solution = array_horizon + [np.array(rend_solution), np.array(rend_equipondere)]
    label_solution = list(data.columns)
    label_solution.append('Solution')
    label_solution.append('Equipondere')
    figAnnualise = ff.create_distplot(array_solution, label_solution, histnorm='probability', show_rug=False, colors=list20Couleurs, bin_size=.5)
    figAnnualise.update_traces(marker={"opacity": 0.1})
    figAnnualise.show()


    # On récupère les poids
    dataSolution = pd.DataFrame({})
    dataSolution["Actifs"] = data.columns
    dataSolution["Poids"] = w_solution

    # On verifie que la contrainte est proche de 0
    print("Contrainte :", contrainte(w_solution))

    # On definit la matrice de correlation
    corr = np.corrcoef([data[data.columns[i]] for i in range(len(data.columns))])
    print(corr[11, 14])



    # On calcul les CVAR

    Strategy(w_solution)
    Strategy(w_equipondere)

    for i in range(len(data.columns)):
        w_1hot = [1 if x==i else 0 for x in range(len(data.columns))]
        print(Strategy(w_1hot))

    w_random = [random.uniform(0, 1) for _ in range(len(data.columns))]
    w_random = [x/sum(w_random) for x in w_random]
    print(Strategy(w_random))
