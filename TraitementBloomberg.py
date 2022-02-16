import numpy
import pandas as pd
import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import pylab as pl
from dash.dependencies import Input, Output
from dash_table import *
import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import openpyxl
import statistics
import seaborn as sns
import statistics as stat

if __name__ == '__main__':

    #liste de 20 couleurs pour différencier les indices ou valeurs
    list20Couleurs = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

    # On charge les données, dans data on a différents fond
    data = pd.read_csv('C:/Users/C00000404/Documents/DossierPartage/Actif/Extract_Bloomberg_sansNaN.csv', encoding='UTF-8', sep=';')
    IndiceMSCIWorldIndex = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='MSCIWorldIndex')
    IndiceMSCIWorldIndex.dropna(inplace=True)
    IndiceMSCIWorldIndex.reset_index(drop=True, inplace=True)
    IndiceEurostoxx50 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='Eurostoxx50')
    IndiceEurostoxx50.dropna(inplace=True)
    IndiceEurostoxx50.reset_index(drop=True, inplace=True)
    IndiceSNP500 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='SNP500')
    IndiceSNP500.dropna(inplace=True)
    IndiceSNP500.reset_index(drop=True, inplace=True)
    IndiceDowJones = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='DowJones')
    IndiceDowJones.dropna(inplace=True)
    IndiceDowJones.reset_index(drop=True, inplace=True)
    IndiceNasdaq = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='Nasdaq')
    IndiceNasdaq.dropna(inplace=True)
    IndiceNasdaq.reset_index(drop=True, inplace=True)
    IndiceNikkei = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='Nikkei')
    IndiceNikkei.dropna(inplace=True)
    IndiceNikkei.reset_index(drop=True, inplace=True)
    IndiceShanghai = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='Shanghai')
    IndiceShanghai.dropna(inplace=True)
    IndiceShanghai.reset_index(drop=True, inplace=True)
    IndiceDax = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='Dax')
    IndiceDax.dropna(inplace=True)
    IndiceDax.reset_index(drop=True, inplace=True)
    IndiceFTSE100 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='FTSE100')
    IndiceFTSE100.dropna(inplace=True)
    IndiceFTSE100.reset_index(drop=True, inplace=True)
    IndiceFTSE250 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='FTSE250')
    IndiceFTSE250.dropna(inplace=True)
    IndiceFTSE250.reset_index(drop=True, inplace=True)
    IndiceCAC40 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='CAC40')
    IndiceCAC40.dropna(inplace=True)
    IndiceCAC40.reset_index(drop=True, inplace=True)
    IndiceIBEX35 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='IBEX35')
    IndiceIBEX35.dropna(inplace=True)
    IndiceIBEX35.reset_index(drop=True, inplace=True)
    IndiceBEL20 = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='BEL20')
    IndiceBEL20.dropna(inplace=True)
    IndiceBEL20.reset_index(drop=True, inplace=True)
    IndiceFTSEItalie = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='FTSEItalie')
    IndiceFTSEItalie.dropna(inplace=True)
    IndiceFTSEItalie.reset_index(drop=True, inplace=True)
    IndiceAEX = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='AEX')
    IndiceAEX.dropna(inplace=True)
    IndiceAEX.reset_index(drop=True, inplace=True)
    IndiceRTS = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='RTS')
    IndiceRTS.dropna(inplace=True)
    IndiceRTS.reset_index(drop=True, inplace=True)
    IndiceSMI = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='SMI')
    IndiceSMI.dropna(inplace=True)
    IndiceSMI.reset_index(drop=True, inplace=True)
    IndiceFTSEAustralie = pd.read_excel("C:/Users/C00000404/Documents/DossierPartage/Actif/Indices.xlsx", index_col=None, sheet_name='FTSEAustralie')
    IndiceFTSEAustralie.dropna(inplace=True)
    IndiceFTSEAustralie.reset_index(drop=True, inplace=True)
    listIndicesData = [IndiceMSCIWorldIndex, IndiceEurostoxx50, IndiceSNP500, IndiceDowJones, IndiceNasdaq, IndiceNikkei, IndiceShanghai, IndiceDax, IndiceFTSE100, IndiceFTSE250, IndiceCAC40, IndiceIBEX35, IndiceBEL20, IndiceFTSEItalie, IndiceAEX, IndiceRTS, IndiceSMI, IndiceFTSEAustralie]
    listIndices = ['IndiceMSCIWorldIndex', 'IndiceEurostoxx50', 'IndiceSNP500', 'IndiceDowJones', 'IndiceNasdaq', 'IndiceNikkei', 'IndiceShanghai', 'IndiceDax', 'IndiceFTSE100', 'IndiceFTSE250', 'IndiceCAC40', 'IndiceIBEX35', 'IndiceBEL20', 'IndiceFTSEItalie', 'IndiceAEX', 'IndiceRTS', 'IndiceSMI', 'IndiceFTSEAustralie']
    for numIndice in range(len(listIndices)):
        listIndicesData[numIndice].columns = ['Date', 'Cours']

    ### --------------------------------------- Les lignes qui suivent servent a effectuer des traitement sur la base ETF
    # for ligne in range(len(dataETF)):
    #     for colonne in range(len(listETF)):
    #         if (dataETF.iloc[ligne, colonne]) == '':
    #             dataETF.iloc[ligne, colonne] = 0
    #         else:
    #             dataETF.iloc[ligne, colonne] = float(str(dataETF.iloc[ligne, colonne]).replace(",", ".").replace(" -   ", "0"))
    # for colonne in listETF:
    #     dataETF = dataETF.astype({colonne: float})
    # for ligne in range(len(dataETF)):
    #     for colonne in range(len(listETF)):
    #         if dataETF.iloc[ligne, colonne] == 0:
    #             for i in range(ligne+1, len(dataETF)):
    #                 if dataETF.iloc[i, colonne] != 0:
    #                     dataETF.iloc[ligne, colonne] = dataETF.iloc[i, colonne]
    #                     break
    #     dataETF.to_csv('C:/Users/C00000404/Documents/DossierPartage/Actif/BaseETF.csv')


    # # On plot les différents indices
    # figCours = px.line(IndiceMSCIWorldIndex, x='Date', y='Cours')
    # for numIndice in range(1, len(listIndices)):
    #     figCours.add_trace(go.Scatter(x=listIndicesData[numIndice]['Date'], y=listIndicesData[numIndice]['Cours'], mode='lines', name=listIndices[numIndice], line=go.scatter.Line(color=list20Couleurs[numIndice])))
    # figCours.show()


    # On créer une liste d'horizons possible
    listHorizon = [i for i in range(1, 21)]
    listHorizon = [annee*365 for annee in listHorizon]

    # # On calcul les rendements journaliers pour les différentes valeurs dans data
    # dataRendements = data.copy()
    # for colonne in range(1, len(dataRendements.columns)):
    #     for ligne in range(len(dataRendements)-1):
    #         dataRendements.iloc[ligne, colonne] = (dataRendements.iloc[ligne, colonne]-dataRendements.iloc[ligne+1, colonne])/dataRendements.iloc[ligne+1, colonne]
    # dicoRendementParHorizon = {}

    # # On calcul et affiche les densité pour différents horizons
    # for horizon in listHorizon:
    #     dataTmp=data.copy()
    #     for colonne in range(1, len(data.columns)):
    #         for ligne in range(len(data)-horizon):
    #             dataTmp.iloc[ligne, colonne] = (dataTmp.iloc[ligne, colonne]-dataTmp.iloc[ligne+horizon, colonne])/dataTmp.iloc[ligne+horizon, colonne]
    #     dicoRendementParHorizon[horizon] = dataTmp
    # horizon = 365
    # dicoRendementParHorizon[horizon].dropna(inplace=True)
    # dicoRendementParHorizon[horizon] = dicoRendementParHorizon[horizon].drop(['Date'], axis=1)
    # hist = [dicoRendementParHorizon[horizon].iloc[:len(dicoRendementParHorizon[horizon]), colonne].to_numpy() for colonne in range(len(dicoRendementParHorizon[horizon].columns))]
    # fig = ff.create_distplot(hist, ['NovaEurope', 'Nasdaqx2', 'TotalEnergies', 'Axa', 'IndiceMondeBloomberg', 'ETF Monde', 'CAC40', 'Eurostoxx50', 'AirLiquide', 'S&P500', 'Nasdaq'], histnorm='probability', show_rug=False)
    # fig.show()

    # On calcul et plot les densité des indices pour différents horizons possibles
    dicoRendementParHorizonIndice = {}
    dicoRendementParHorizonIndiceAnnualise = {}
    for horizon in listHorizon:
        dicoRendementParHorizonIndice[horizon] = []
        dicoRendementParHorizonIndiceAnnualise[horizon] = []
        for indice in listIndicesData:
            dicoRendementParHorizonIndice[horizon].append(np.array([(indice.iloc[ligne, 1]-indice.iloc[ligne+horizon, 1])/indice.iloc[ligne+horizon, 1] for ligne in range(len(indice)-horizon)]))
            dicoRendementParHorizonIndiceAnnualise[horizon].append(np.array([((1 + (indice.iloc[ligne, 1] - indice.iloc[ligne + horizon, 1]) / indice.iloc[ligne + horizon, 1])**(365/horizon)) -1 for ligne in range(len(indice) - horizon)]))

    # On choisit un horizon et on trace les densités sur un même graph pour les rendement annualisé ou non
    horizon = 365*10
    # fig = ff.create_distplot(dicoRendementParHorizonIndice[horizon], listIndices, histnorm='probability', show_rug=False, colors=list20Couleurs)
    # fig.update_traces(marker={"opacity": 0.1})
    # fig.show()
    figAnnualise = ff.create_distplot(dicoRendementParHorizonIndiceAnnualise[horizon], listIndices, histnorm='probability', show_rug=False, colors=list20Couleurs, bin_size=.5)
    figAnnualise.update_traces(marker={"opacity": 0.1})
    figAnnualise.show()
    # On s'interesse ici a la var, quantile alpha pourcent. On créer une liste de quantile, on les trace eventuellement et on en fait un bdd triée
    alpha = 0.05
    listQuantileIndices = [np.quantile(dicoRendementParHorizonIndiceAnnualise[horizon][numIndice], alpha) for numIndice in range(len(listIndicesData))]
    # permet de tracer les quantiles comme lignes verticales
    # listTraceQuantileIndices = [figAnnualise.add_trace(go.Scatter(x=[listQuantileIndices[numIndice], listQuantileIndices[numIndice]], y=[0, 10], mode="lines", line=go.scatter.Line(color=list20Couleurs[numIndice]), showlegend=False)) for numIndice in range(len(listIndicesData))]
    # On créer une base de donnée avec les quantiles et on la trie pour voir les meilleurs éléments
    listCVar = [stat.mean([dicoRendementParHorizonIndiceAnnualise[horizon][numIndice][rendement] for rendement in range(len(dicoRendementParHorizonIndiceAnnualise[horizon][numIndice])) if dicoRendementParHorizonIndiceAnnualise[horizon][numIndice][rendement]<=listQuantileIndices[numIndice]]) for numIndice in range(len(listIndicesData))]
    dataQuantile = pd.DataFrame({'Indices': listIndices, 'Var': listQuantileIndices, 'CVar': listCVar}).sort_values(by='CVar', ascending=False)

    # On print le cac avec sa densité
    from scipy.stats import gaussian_kde
    CAC_rendAnnualise = list(dicoRendementParHorizonIndiceAnnualise[365*10][10])
    CAC_densite = gaussian_kde(CAC_rendAnnualise)
    x = np.linspace(-0.5, 0.5, 2000)
    print(list(x))
    print([CAC_densite(i)[0] for i in x])
    CAC_airNegative = sum([1 for x in CAC_rendAnnualise if x<0])/len(CAC_rendAnnualise)
    CAC_aircentrale = sum([1 for x in CAC_rendAnnualise if x>=0.02 and x<0.085])/len(CAC_rendAnnualise)
    CAC_airdroite = sum([1 for x in CAC_rendAnnualise if x>=0.1])/len(CAC_rendAnnualise)
    CAC_moyenne = sum(CAC_rendAnnualise)/len(CAC_rendAnnualise)


    # # Markovitz
    # listRendementsIndices = [(1+((indice['Cours'][0]-indice['Cours'][len(indice)-1])/indice['Cours'][len(indice)-1]))**(365.25/len(indice))-1 for indice in listIndicesData]
    # listVolatilityIndices = [statistics.stdev([(indice.iloc[ligne, 1]-indice.iloc[ligne+1, 1])/indice.iloc[ligne+1, 1] for ligne in range(len(indice)-1)]) for indice in listIndicesData]
    # R = np.array(listRendementsIndices)
    # omega = np.cov(np.array([[(indice.iloc[ligne, 1]-indice.iloc[ligne+1, 1])/indice.iloc[ligne+1, 1] for ligne in range(min([len(i) for i in listIndicesData])-1)] for indice in listIndicesData]), bias=True)
    # oneT = np.ones((1, len(listIndices)), dtype=np.int32)
    # A = oneT.dot(np.linalg.inv(omega).dot(R))
    # B = R.transpose().dot(np.linalg.inv(omega).dot(oneT.transpose()))
    # C = oneT.dot(np.linalg.inv(omega).dot(oneT.transpose()))
    # D = B*C-A*A
    # g = (B*np.linalg.inv(omega).dot(oneT.transpose())-A*np.linalg.inv(omega).dot(R))/D
    # h = (C*np.linalg.inv(omega).dot(R)-A*np.linalg.inv(omega).dot(oneT.transpose()))/D
    # Ep = [0.001*i for i in range(-500, 500)]
    # Sigma = (C*[i**2 for i in Ep]-2*A*Ep+B)/D
    # Sigma = Sigma[0]
    # figMarkovitz = px.scatter(x=listVolatilityIndices, y=listRendementsIndices, color=listIndices, hover_name=listIndices)
    # figMarkovitz.add_trace(go.Scatter(x=Sigma, y=Ep, mode='lines', name='Frontière Efficiente'))
    # figMarkovitz.show()



    #
    # nombreAnneeHistorique = 15
    # # Markovitz
    # # On definit une list qui pour chaque indice prend une liste des rendements annuel sur 15ans (par exemple), avec un pas d'un an, ie années apres années
    # listlistRendements = [[(indice.iloc[ligne, 1]-indice.iloc[ligne+365, 1])/indice.iloc[ligne+365, 1] for ligne in range(0, nombreAnneeHistorique*365, 365)] for indice in listIndicesData[:5]]
    # # On definit la liste des rendement moyen sur 15ans de chaque indices
    # listRendementsIndices = [sum(rend)/len(rend) for rend in listlistRendements]
    # # On definit la volatilisté des rendements annuel (et non journalier) pour chaque indice
    # listVolatilityIndices = [statistics.stdev(rend) for rend in listlistRendements]
    # # On calcul les paramètre de markovitz
    # R = np.array(listRendementsIndices)
    # omega = np.cov(np.array(listlistRendements))
    # for i in range(len(listlistRendements)):
    #     for j in range(len(listlistRendements)):
    #         omega[i][j]=round(omega[i][j], 20)
    # omegaI = np.linalg.inv(omega)
    # for i in range(len(listlistRendements)):
    #     for j in range(len(listlistRendements)):
    #         if abs(omega[i][j]-np.linalg.inv(omegaI)[i][j]) > 0.001: print("probleme")
    # oneT = np.ones((1, len(listlistRendements)), dtype=np.int32)
    # A = R.transpose().dot(omegaI.dot(oneT.transpose()))[0]
    # B = R.transpose().dot(omegaI.dot(R))
    # C = oneT.dot(omegaI.dot(oneT.transpose()))
    # D = B*C-A*A
    # g = (B*omegaI.dot(oneT.transpose())-A*omegaI.dot(R))/D
    # h = (C*omegaI.dot(R)-A*omegaI.dot(oneT.transpose()))/D
    # Ep = [0.001*i for i in range(-500, 500)]
    # Sigma = [((C*i**2-2*A*i+B)/D)[0][0] for i in Ep]
    # figMarkovitz = px.scatter(x=listVolatilityIndices, y=listRendementsIndices, color=listIndices[:5], hover_name=listIndices[:5])
    # figMarkovitz.add_trace(go.Scatter(x=Sigma, y=Ep, mode='lines', name='Frontière Efficiente'))
    # figMarkovitz.show()
    #
    # # Mettre les conditions d'interdiction de vente a perte



    # Test Maximums historique
    listIndices = ['IndiceMSCIWorldIndex', 'IndiceEurostoxx50', 'IndiceSNP500', 'IndiceDowJones', 'IndiceNasdaq', 'IndiceNikkei', 'IndiceShanghai', 'IndiceDax', 'IndiceFTSE100', 'IndiceFTSE250', 'IndiceCAC40', 'IndiceIBEX35', 'IndiceBEL20', 'IndiceFTSEItalie', 'IndiceAEX', 'IndiceRTS', 'IndiceSMI', 'IndiceFTSEAustralie']
    numeroIndice = 2
    MH_methode = "L"
    for numeroIndice in range(len(listIndices)):
        x = listIndicesData[numeroIndice]['Date']
        if MH_methode == "L":
            # Par lissage
            MH_cours = [val for i, val in enumerate(listIndicesData[numeroIndice]['Cours'])]
            LissagePuissance = 5
            LissageLargeur = 120
            for iteration in range(LissagePuissance):
                MH_cours = [np.mean(MH_cours[i - LissageLargeur:i + LissageLargeur]) for i in range(len(MH_cours[LissageLargeur:-LissageLargeur]))]
                x = x[LissageLargeur:-LissageLargeur]
        elif MH_methode == "E":
            # Par echantillonage
            echantillonage = 120
            MH_cours = [val for i, val in enumerate(listIndicesData[numeroIndice]['Cours']) if i%echantillonage==0]
            x = [i for i in x if i%echantillonage==0]
        MH_compteurMaxHisto = [1 if not [1 for valFuture in MH_cours[i+1:] if valFuture>val] else 0 for i,val in enumerate(MH_cours)]
        x = list(x)
        x.reverse()
        MH_cours.reverse()
        MH_compteurMaxHisto.reverse()
        MH_pourcentage = sum(MH_compteurMaxHisto)/len(MH_compteurMaxHisto)
        print(f'Le cours {listIndices[numeroIndice]} est {round(100*MH_pourcentage, 2)}% du temps a son maximum historique.')
        plt.figure()
        plt.title(listIndices[numeroIndice])
        plt.plot(x, MH_cours)
        plt.plot(x, [MH_compteurMaxHisto[i]*MH_cours[i] for i in range(len(MH_cours))])






    # Strategie buy-the-dip
    listIndices = ['IndiceMSCIWorldIndex', 'IndiceEurostoxx50', 'IndiceSNP500', 'IndiceDowJones', 'IndiceNasdaq', 'IndiceNikkei', 'IndiceShanghai', 'IndiceDax', 'IndiceFTSE100', 'IndiceFTSE250', 'IndiceCAC40', 'IndiceIBEX35', 'IndiceBEL20', 'IndiceFTSEItalie', 'IndiceAEX', 'IndiceRTS', 'IndiceSMI', 'IndiceFTSEAustralie']
    numeroIndice = 10
    montantInvestitParMois = 100
    nbrJourOuvre = 254
    pourcentageDip = -0.2

    strategiePassive = [0 for i,val in enumerate(listIndicesData[numeroIndice]['Cours'])] # Liste d'investissemnt de 100€ par mois
    for i,val in enumerate(listIndicesData[numeroIndice]['Cours']):
        if i == 0:
            strategiePassive[i] = montantInvestitParMois
        elif i % 30 == 0: # Si on se trouve au debut d'un mois
            strategiePassive[i] = strategiePassive[i-1] * (1 + (listIndicesData[numeroIndice]['Cours'][i]-listIndicesData[numeroIndice]['Cours'][i-1])/listIndicesData[numeroIndice]['Cours'][i-1]) + montantInvestitParMois
        else:
            strategiePassive[i] = strategiePassive[i-1] * (1 + (listIndicesData[numeroIndice]['Cours'][i]-listIndicesData[numeroIndice]['Cours'][i-1])/listIndicesData[numeroIndice]['Cours'][i-1])

    strategieBTD = [0 for i,val in enumerate(listIndicesData[numeroIndice]['Cours'])] # Liste d'investissemnt de 100€ par mois
    liquiditeEnStock = 0
    indiceLiquidite = 0
    for i,val in enumerate(listIndicesData[numeroIndice]['Cours']):
        if i % 30 == 0: # Si on se trouve au debut d'un mois
            liquiditeEnStock += 1
            if indiceLiquidite==0:
                indiceLiquidite = i
        if i>0 and liquiditeEnStock>0 and (listIndicesData[numeroIndice]['Cours'][i]-listIndicesData[numeroIndice]['Cours'][indiceLiquidite])/listIndicesData[numeroIndice]['Cours'][indiceLiquidite] <= pourcentageDip:
            strategieBTD[i] = strategieBTD[i-1] * (1 + (listIndicesData[numeroIndice]['Cours'][i]-listIndicesData[numeroIndice]['Cours'][i-1])/listIndicesData[numeroIndice]['Cours'][i-1]) + montantInvestitParMois * liquiditeEnStock
            liquiditeEnStock = 0
            indiceLiquidite = 0
        elif i>0:
            strategieBTD[i] = strategieBTD[i - 1] * (1 + (listIndicesData[numeroIndice]['Cours'][i] - listIndicesData[numeroIndice]['Cours'][i - 1]) / listIndicesData[numeroIndice]['Cours'][i - 1])
        # print(i, liquiditeEnStock, indiceLiquidite)

    rendementAnnualiseStrategiePassive = (1+(strategiePassive[-1]-100)/100)**(nbrJourOuvre/len(listIndicesData[numeroIndice]))-1
    rendementAnnualiseStrategieBTD = (1+(strategieBTD[-1]-100)/100)**(nbrJourOuvre/len(listIndicesData[numeroIndice]))-1
    performanceActive = len([0 for i in range(len(listIndicesData[numeroIndice])) if strategieBTD[i]>strategiePassive[i]])/len(listIndicesData[numeroIndice])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x=listIndicesData[numeroIndice]['Date'].sort_values()
    ax1.plot(x, strategiePassive, 'blue')
    ax1.plot(x, strategieBTD, 'red')
    ax2.plot(x, listIndicesData[numeroIndice]['Cours'], 'grey')
    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y1 data', color='blue')
    ax2.set_ylabel('Y2 data', color='grey')
    plt.show()

    print("Rendement Passif : ", rendementAnnualiseStrategiePassive)
    print("Rendement BTD : ", rendementAnnualiseStrategieBTD)
    print("Pourcenage de jour où la gestion active est rentable : ", performanceActive)
















