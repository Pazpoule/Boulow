import numpy
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from matplotlib import pyplot
from plotly.subplots import make_subplots
import webbrowser
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import openpyxl
import statistics
import seaborn as sns
import math
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import openpyxl
from openpyxl import Workbook, load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

adresse = 'C:/Users/pnguyen/Desktop/ROOT/Proportionnel/'
excel = load_workbook(adresse + "template.xlsx")


if __name__ == '__main__':

    print("DEBUT ---------------------------------------------------------------")
    # ------------------- Import DATA -----------------------------------------------
    Import_RevenusRecentsMoyen_PLME = pd.read_excel(adresse+"RevenusRecentsMoyen_PLME.xlsx", index_col=None, sheet_name='Feuil1')
    RevenusReels = Import_RevenusRecentsMoyen_PLME[Import_RevenusRecentsMoyen_PLME['revenus_moyens']>=1527]
    RevenusReels.reset_index(inplace=True)

    # ------------------- Choix paramètre -----------------------------------------
    rendementDuPoint = 0.0620
    PASS = 41136
    RevenuParClasse = [0, 4453, 5712, 6570, 7054, 26580, 49280, 57850, 66400, 83060, 103180, 123300]
    CotisationParClasse = [0, 382, 764, 1145, 1527, 3055, 4582, 7637, 10692, 16802, 18329, 19857]
    dureeCotisation = 42
    revenuMaxChoisi = 1000000
    revenuMaxOptimisation = 130000
    plafondnbr = 4
    plafondRevenu = plafondnbr * PASS
    valeurDeServicePoint = 2.63
    borne = 130000

    # ------------------- Initialisation des variables -------------------------------------------
    dicoRevenu = {'Total': [], 'PL': [], 'ME': []}
    dicoCotisation = {'Total': [], 'PL': [], 'ME': []}
    dicoNbrCotParClasse = {'Total': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'PL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'ME': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    dicoTauxRemplacement = {'Total': [], 'PL': [], 'ME': []}
    dicoCotisation_Classes = {'Total': [], 'PL': [], 'ME': []}
    valeurAchatPoint = valeurDeServicePoint / rendementDuPoint

    print("TRANCHAGE PAR CLASSE ---------------------------------------------------")
    # --------------------- Tranchage par classe ---------- Compte le nombre / cotisation / taux de remplacement ------- par classe
    for index in RevenusReels.index:
        PLME = RevenusReels.loc[index, 'PL_ME']
        revenu = RevenusReels.loc[index, 'revenus_moyens']
        dicoRevenu['Total'].append(revenu)
        if PLME == 'PL':
            dicoRevenu['PL'].append(revenu)
        else:
            dicoRevenu['ME'].append(revenu)
        for i in range(len(RevenuParClasse)-1):
            if revenu>=RevenuParClasse[i] and revenu<RevenuParClasse[i+1]:
                dicoCotisation['Total'].append(CotisationParClasse[i])
                dicoNbrCotParClasse['Total'][i] += 1
                dicoTauxRemplacement['Total'].append(dureeCotisation*rendementDuPoint*CotisationParClasse[i]/revenu)
                if PLME == 'PL':
                    dicoCotisation['PL'].append(CotisationParClasse[i])
                    dicoNbrCotParClasse['PL'][i] += 1
                    dicoTauxRemplacement['PL'].append(dureeCotisation * rendementDuPoint * CotisationParClasse[i] / revenu)
                else:
                    dicoCotisation['ME'].append(CotisationParClasse[i])
                    dicoNbrCotParClasse['ME'][i] += 1
                    dicoTauxRemplacement['PL'].append(dureeCotisation * rendementDuPoint * CotisationParClasse[i] / revenu)
        if revenu >= RevenuParClasse[-1]:
            dicoCotisation['Total'].append(CotisationParClasse[-1])
            dicoNbrCotParClasse['Total'][-1] += 1
            dicoTauxRemplacement['Total'].append(dureeCotisation * rendementDuPoint * CotisationParClasse[-1] / revenu)
            if PLME == 'PL':
                dicoCotisation['PL'].append(CotisationParClasse[-1])
                dicoNbrCotParClasse['PL'][-1] += 1
                dicoTauxRemplacement['PL'].append(dureeCotisation*rendementDuPoint*CotisationParClasse[-1]/revenu)
            else:
                dicoCotisation['ME'].append(CotisationParClasse[-1])
                dicoNbrCotParClasse['ME'][-1] += 1
                dicoTauxRemplacement['ME'].append(dureeCotisation * rendementDuPoint * CotisationParClasse[-1] / revenu)

    # -------------------- Idem pour une distribution de revenu uniforme
    for revenu in range(revenuMaxChoisi):
        for i in range(len(RevenuParClasse) - 1):
            if revenu >= RevenuParClasse[i] and revenu < RevenuParClasse[i + 1]:
                dicoCotisation_Classes['Total'].append(CotisationParClasse[i])
        if revenu >= RevenuParClasse[-1]:
            dicoCotisation_Classes['Total'].append(CotisationParClasse[-1])


    # --------------------- Statistique -----------------------------------
    plt.figure(figsize=[16, 9])
    plt.title("Nombre de cotisants par classe")
    plt.plot(dicoNbrCotParClasse['Total'])
    plt.figure(figsize=[16, 9])
    plt.title("Cotisation recus par classe")
    plt.plot(np.array(dicoNbrCotParClasse['Total'])*np.array(CotisationParClasse))
    print("Nombre de cotisants, et masse de cotisations", dicoNbrCotParClasse['Total'], list(np.array(dicoNbrCotParClasse['Total'])*np.array(CotisationParClasse)))
    plt.figure(figsize=[16, 9])
    plt.title("Densité de taux de Remplacement")
    sns.distplot(dicoTauxRemplacement['Total'], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3})
    print('Taux de remplacement moyen:', np.mean(dicoTauxRemplacement['Total']))
    # ----------- Export de la densité du taux de remplacement
    listTauxRemplacement = []
    for i in range(len(dicoTauxRemplacement['Total'])):
        listTauxRemplacement.append(round(dicoTauxRemplacement['Total'][i], 2))
    OutputTauxRemplacement = pd.DataFrame(pd.DataFrame(listTauxRemplacement).value_counts(), columns=['frequence']).reset_index().sort_values(by=0)
    [excel['TauxRemplacement'].append(row) for row in dataframe_to_rows(OutputTauxRemplacement, index=False, header=False)]


    print("OPTIMISATION -----------------------------------------------------------")
    # --------------------------- Modelisation distance au ADH -------------------------------
    def distanceADH(taux):
        absDistances = []
        for index in RevenusReels.index:
            revenu = RevenusReels.loc[index, 'revenus_moyens']
            if revenu < PASS:
                absDistances.append(abs(taux[0]*revenu - dicoCotisation['Total'][index]))
            elif revenu >= PASS and revenu < plafondRevenu:
                absDistances.append(abs(taux[0]*PASS + taux[1]*(revenu-PASS) - dicoCotisation['Total'][index]))
        return sum(absDistances)
    resADH = minimize(distanceADH, [0.06, 0.14], method='Nelder-Mead').x
    # -------------------- Modelisation distance aux classes ----------------------------------
    def distanceClasses(taux):
        absDistances = []
        for revenu in range(revenuMaxOptimisation):
            if revenu < PASS:
                absDistances.append(abs(taux[0]*revenu - dicoCotisation_Classes['Total'][revenu]))
            elif revenu >= PASS and revenu < plafondRevenu:
                absDistances.append(abs(taux[0]*PASS + taux[1]*(revenu-PASS) - dicoCotisation_Classes['Total'][revenu]))
        return sum(absDistances)
    resClasses = minimize(distanceClasses, [0.06, 0.14], method='Nelder-Mead').x
    # --------------------------- Modelisation distance au taux de remplacement -------------------------------
    tauxCible = 0.3
    def distanceRemplacement(taux):
        absDistances = []
        tauxRemp = []
        for index in RevenusReels.index:
            revenu = RevenusReels.loc[index, 'revenus_moyens']
            if revenu <= revenuMaxOptimisation:
                if revenu < PASS:
                    absDistances.append(abs(taux[0] - dicoCotisation['Total'][index]/revenu))
                    tauxRemp.append(dureeCotisation*rendementDuPoint*taux[0])
                elif revenu >= PASS and revenu < plafondRevenu:
                    absDistances.append(abs((taux[0] * PASS + taux[1] * (revenu - PASS))/revenu - tauxCible/(dureeCotisation*rendementDuPoint)))
                    tauxRemp.append(dureeCotisation*rendementDuPoint*(taux[0] * PASS + taux[1] * (revenu - PASS))/revenu)
        return sum(absDistances)/len(absDistances) + abs(np.mean(tauxRemp) - tauxCible)/20
    resCible = minimize(distanceRemplacement, [0.06, 0.14], method='Nelder-Mead').x

    resADH = [0.09, 0.22]

    print("APPLICATION AUX MODELES ---------------------------------------------------------")
    # ----------------------------- Application des taux optimaux au modèles
    dicoCotisationProportionnelle = {"resADH": [], "resClasses": [], "resCible": []}
    dicoTauxRemplacement = {"resADH": [], "resClasses": [], "resCible": []}
    for res in ["resADH", "resClasses", "resCible"]:
        res = "resADH"
        resTaux = [0.09, 0.22]
        # resTaux = list(resADH * (res=="resADH") + resClasses * (res=="resClasses") + resCible * (res=="resCible"))
        # res = "resADH"
        # dicoCotisationProportionnelle[res] = []
        # resTaux = [0.08, 0.20]
        # -------------------------------- Calcul du taux proportionnelle et taux de remplacement
        for index in RevenusReels.index:
            revenu = RevenusReels.loc[index, 'revenus_moyens']
            if revenu < PASS:
                dicoCotisationProportionnelle[res].append(resTaux[0] * revenu)
                dicoTauxRemplacement[res].append(dureeCotisation * rendementDuPoint * resTaux[0])
            elif revenu >= PASS and revenu < plafondRevenu:
                dicoCotisationProportionnelle[res].append(resTaux[0] * PASS + resTaux[1] * (revenu - PASS))
                dicoTauxRemplacement[res].append(dureeCotisation * rendementDuPoint * (resTaux[0] * PASS + resTaux[1] * (revenu - PASS))/revenu)
            else:
                dicoCotisationProportionnelle[res].append(resTaux[0] * PASS + resTaux[1]*(plafondRevenu-PASS))
                dicoTauxRemplacement[res].append(dureeCotisation * rendementDuPoint * (resTaux[0] * PASS + resTaux[1]*(plafondRevenu-PASS))/revenu)

        # pd.DataFrame(dicoCotisationProportionnelle[res]).to_csv(adresse + "cotisation_arrondi_unite.csv", index=False)
        pd.DataFrame(list(dicoCotisation['Total'])).to_csv(adresse + "cotis_class.csv", index=False)

        listEcartCotisation = [dicoCotisationProportionnelle[res][i] - dicoCotisation['Total'][i] for i in range(len(RevenusReels))]
        listEcartCotisation = sorted(listEcartCotisation)
        listEcartCotisation_arrondi = []
        listEcartCotisation_borne = []
        for i in range(len(listEcartCotisation)):
            listEcartCotisation_arrondi.append(round(listEcartCotisation[i]/100, 0)*100)
        for i in range(len(listEcartCotisation[:borne])):
            listEcartCotisation_borne.append(round(listEcartCotisation[i]/100, 0)*100)

        dicoTauxSelonRendementPoint = {}
        OutputCotisations = pd.DataFrame({"Revenu Réel":dicoRevenu['Total'], "Cotisation par classe":dicoCotisation['Total'], "Cotisation proportonnelle":dicoCotisationProportionnelle[res], "Nombre de points": [round(dicoCotisationProportionnelle[res][i]/valeurAchatPoint, 0) for i in range(len(dicoCotisationProportionnelle[res]))]})
        OutputProjectionPoint = pd.DataFrame({"Revenu Réel":OutputCotisations["Revenu Réel"], "Nombre de points": OutputCotisations["Nombre de points"]})
        for rendementPoint in [0.062, 0.059, 0.056, 0.053, 0.05]:
            VA = valeurDeServicePoint / rendementPoint
            OutputProjectionPoint[f'Cotisation pour r={round(rendementPoint*100, 1)}%'] = [OutputProjectionPoint.loc[i, "Nombre de points"]*VA for i in range(len(OutputProjectionPoint))]
            def trouverTauxSelonRendementPoint(taux):
                absDistances = []
                for index in OutputProjectionPoint.index:
                    revenu = OutputProjectionPoint.loc[index, 'Revenu Réel']
                    if revenu < PASS:
                        absDistances.append(abs(taux[0] * revenu - OutputProjectionPoint.loc[index, f'Cotisation pour r={round(rendementPoint*100, 1)}%']))
                    elif revenu >= PASS and revenu < plafondRevenu:
                        absDistances.append(abs(taux[0] * PASS + taux[1] * (revenu - PASS) - OutputProjectionPoint.loc[index, f'Cotisation pour r={round(rendementPoint*100, 1)}%']))
                return sum(absDistances)
            dicoTauxSelonRendementPoint[rendementPoint] = list(minimize(trouverTauxSelonRendementPoint, resTaux, method='Nelder-Mead').x)

        # -------------------------------- Résultats
        OutputEcartCotisation = pd.DataFrame(pd.DataFrame(listEcartCotisation_arrondi).value_counts(), columns=['frequence']).reset_index().sort_values(by=0)
        OutputEcartCotisation_borne = pd.DataFrame(pd.DataFrame(listEcartCotisation_borne).value_counts(), columns=['frequence']).reset_index().sort_values(by=0)

        # Affichage
        print(f"Optimum {res}:", resTaux)
        plt.figure(figsize=[16, 9])
        plt.title(f"Graph de comparaison des cotisation par classe et proportionnelle {res}")
        plt.plot(dicoRevenu['Total'], dicoCotisation['Total'])
        plt.plot(dicoRevenu['Total'], dicoCotisationProportionnelle[res])
        plt.figure(figsize=[16, 9])
        plt.title(f'Ecart de cotisations {res}')
        sns.distplot(listEcartCotisation, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
        print(f'Ecart de cotiation moyen {res}: ', np.mean(listEcartCotisation))
        print(f'Ecart de cotiation moyen Borné {res}: ', np.mean(listEcartCotisation[:borne]))
        print(f'Taux de remplacement moyen {res}:', np.mean(dicoTauxRemplacement[res]))

        # --------------------------------------- Export excel
        excel['StatsMéthodes'].append([res, np.mean(dicoTauxRemplacement[res]), np.mean(listEcartCotisation), np.median(listEcartCotisation), min(listEcartCotisation), max(listEcartCotisation), len([i for i in listEcartCotisation if i>=0])/len(listEcartCotisation), np.mean(listEcartCotisation[:borne])])
        excel['Exemple'].append(resTaux)
        [excel[f'EcartCotisation{res}'].append(row) for row in dataframe_to_rows(OutputEcartCotisation, index=False, header=False)]
        [excel[f'EcartCotisation{res}_cut'].append(row) for row in dataframe_to_rows(OutputEcartCotisation_borne, index=False, header=False)]
        [excel[f'Cotisations{res}'].append(row) for row in dataframe_to_rows(OutputCotisations, index=False, header=False)]
        [excel[f'TxSelonRendementPoint{res}'].append(row) for row in dataframe_to_rows(OutputProjectionPoint, index=False, header=True)]
        [excel['TauxSelonRendementPoint'].append([res, i] + dicoTauxSelonRendementPoint[i]) for i in dicoTauxSelonRendementPoint]

    excel.save(adresse + f"output_{plafondnbr}PASS_choixFinal.xlsx")

print("FIN ------------------------------------------------------")





