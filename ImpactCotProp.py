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
# import plotly.io as pio
# pio.renderers.default = "browser"

def plotDistrib(dico, normalize=True, titre="Distribution"):
    plt.figure(figsize=[16, 9])
    plt.title(titre)
    for index in dico:
        distrib = pd.DataFrame(dico[index].value_counts())
        if normalize: distrib = distrib / distrib.sum()
        distrib.sort_index(inplace=True)
        plt.plot(distrib, label=str(index))
        plt.legend()

adresse = "C:/Users/pnguyen/Desktop/ROOT/Proportionnel/"
data = pd.read_csv(adresse + 'revenus_cotisation_pl.csv', sep=';', encoding='latin-1')
inflation = pd.read_csv(adresse + 'inflation_historique.csv')
data = data.merge(inflation, left_on='ANNEE_COTISATION', right_on='annee').drop_duplicates()
data.drop('ANNEE_COTISATION', axis=1, inplace=True)

inflationRevenus = False # TODO à choisir avant execution --------------------------------------------
if inflationRevenus:
    data["MONTANT_REVENU"] = data["MONTANT_REVENU"] * (1.03)**3


sample = data.sample(1000)
plt.scatter(sample['annee'], sample['MONTANT_REVENU'])


dataAnneeRecentes = data[data["annee"].isin([2019, 2020, 2021])]
tauxRevenuNonPresent = len(dataAnneeRecentes[dataAnneeRecentes["MONTANT_REVENU"].isna()])/len(dataAnneeRecentes)

dataAnneeRecentesHorsNA = dataAnneeRecentes[dataAnneeRecentes["MONTANT_REVENU"]>0]
dataRevenuMoyen = dataAnneeRecentesHorsNA.groupby(by=["NUM_ADHERENT_FONCT"]).mean()
dataRevenuMoyen.reset_index(inplace=True)
dataRevenuMoyen = dataRevenuMoyen.drop(["NUM_ADHERENT", "annee"], axis=1)
dataRevenuMoyen.sort_values(by=["MONTANT_REVENU"], inplace=True, ignore_index=True)
plotDistrib({"DistributionRevenuMoyen": dataRevenuMoyen[dataRevenuMoyen["MONTANT_REVENU"]<400000]["MONTANT_REVENU"]})

plt.plot(dataRevenuMoyen["MONTANT_REVENU"], dataRevenuMoyen["Cotisation_appele"])



# ------------------- Choix paramètre -----------------------------------------
rendementDuPoint = 0.0620
PASS = 41136
RevenuParClasse = [0, 4453, 5712, 6570, 7054, 26580, 49280, 57850, 66400, 83060, 103180, 123300]
CotisationParClasse = [0, 382, 764, 1145, 1527, 3055, 4582, 7637, 10692, 16802, 18329, 19857]
classes = ['A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
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
dicoClasses = {'Total': [], 'PL': [], 'ME': []}
dicoNbrCotParClasse = {'Total': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'PL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'ME': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
dicoTauxRemplacement = {'Total': [], 'PL': [], 'ME': []}
dicoCotisation_Classes = {'Total': [], 'PL': [], 'ME': []}
valeurAchatPoint = valeurDeServicePoint / rendementDuPoint

print("TRANCHAGE PAR CLASSE ---------------------------------------------------")
# --------------------- Tranchage par classe ---------- Compte le nombre / cotisation / taux de remplacement ------- par classe
for index in dataRevenuMoyen.index:
    revenu = dataRevenuMoyen.loc[index, 'MONTANT_REVENU']
    dicoRevenu['Total'].append(revenu)
    for i in range(len(RevenuParClasse) - 1):
        if revenu >= RevenuParClasse[i] and revenu < RevenuParClasse[i + 1]:
            dicoCotisation['Total'].append(CotisationParClasse[i])
            dicoNbrCotParClasse['Total'][i] += 1
            dicoTauxRemplacement['Total'].append(dureeCotisation * rendementDuPoint * CotisationParClasse[i] / revenu)
            dicoClasses['Total'].append(classes[i])
    if revenu >= RevenuParClasse[-1]:
        dicoCotisation['Total'].append(CotisationParClasse[-1])
        dicoNbrCotParClasse['Total'][-1] += 1
        dicoTauxRemplacement['Total'].append(dureeCotisation * rendementDuPoint * CotisationParClasse[-1] / revenu)
        dicoClasses['Total'].append(classes[-1])

print("APPLICATION AUX MODELES ---------------------------------------------------------")
# ----------------------------- Application des taux optimaux au modèles
dicoCotisationProportionnelle = {"Rendement=6.2%": [], "Rendement=5.6%": []}
dicoTauxRemplacement = {"Rendement=6.2%": [], "Rendement=5.6%": []}
dicoTranche = {"Rendement=6.2%": [], "Rendement=5.6%": []}
for rendement in ["Rendement=6.2%", "Rendement=5.6%"]:
    taux = [0.08, 0.20] * (rendement == "Rendement=6.2%") + [0.09, 0.22] * (rendement == "Rendement=5.6%")
    # -------------------------------- Calcul du taux proportionnelle et taux de remplacement
    for index in dataRevenuMoyen.index:
        revenu = dataRevenuMoyen.loc[index, 'MONTANT_REVENU']
        if revenu < PASS:
            dicoCotisationProportionnelle[rendement].append(taux[0] * revenu)
            dicoTauxRemplacement[rendement].append(dureeCotisation * rendementDuPoint * taux[0])
            dicoTranche[rendement].append(1)
        elif revenu >= PASS and revenu < plafondRevenu:
            dicoCotisationProportionnelle[rendement].append(taux[0] * PASS + taux[1] * (revenu - PASS))
            dicoTauxRemplacement[rendement].append(dureeCotisation * rendementDuPoint * (taux[0] * PASS + taux[1] * (revenu - PASS)) / revenu)
            dicoTranche[rendement].append(2)
        else:
            dicoCotisationProportionnelle[rendement].append(taux[0] * PASS + taux[1] * (plafondRevenu - PASS))
            dicoTauxRemplacement[rendement].append(dureeCotisation * rendementDuPoint * (taux[0] * PASS + taux[1] * (plafondRevenu - PASS)) / revenu)
            dicoTranche[rendement].append(3)
            
# On aggrege les résultats a la base
dataRevenuMoyen["classe"] = dicoClasses['Total']
dataRevenuMoyen['tranche'] = dicoTranche["Rendement=6.2%"] # 6.2 ou 5.6 sont les mêmes tranches
dataRevenuMoyen["cotisation_classe"] = dicoCotisation['Total']
dataRevenuMoyen["Rendement=6.2%_cotisation"] = dicoCotisationProportionnelle["Rendement=6.2%"]
dataRevenuMoyen["Rendement=5.6%_cotisation"] = dicoCotisationProportionnelle["Rendement=5.6%"]

# On plot l'escalier
plt.figure(figsize=[16, 9])
plt.plot(dataRevenuMoyen["MONTANT_REVENU"], dataRevenuMoyen["cotisation_classe"], label="Par Classe")
plt.plot(dataRevenuMoyen["MONTANT_REVENU"], dataRevenuMoyen["Rendement=6.2%_cotisation"], label="rendement 6.2%")
plt.plot(dataRevenuMoyen["MONTANT_REVENU"], dataRevenuMoyen["Rendement=5.6%_cotisation"], label="rendement 5.6%")
plt.legend()

# On calcul les cotisations pour chaque méthode
cotisationTotalClasse = sum(dataRevenuMoyen["cotisation_classe"])
cotisationTotal62 = sum(dataRevenuMoyen["Rendement=6.2%_cotisation"])
cotisationTotal62PremiereTranche = sum(dataRevenuMoyen[dataRevenuMoyen["MONTANT_REVENU"]<PASS]["Rendement=6.2%_cotisation"])
cotisationTotal62SecondeTranche = sum(dataRevenuMoyen[dataRevenuMoyen["MONTANT_REVENU"]>=PASS]["Rendement=6.2%_cotisation"])
cotisationTotal62SecondeTranche_plafond = sum(dataRevenuMoyen[(dataRevenuMoyen["MONTANT_REVENU"]>=PASS) & (dataRevenuMoyen["MONTANT_REVENU"]<plafondRevenu)]["Rendement=6.2%_cotisation"])
cotisationTotal62TroisiemeTranche = sum(dataRevenuMoyen[(dataRevenuMoyen["MONTANT_REVENU"]>=plafondRevenu)]["Rendement=6.2%_cotisation"])

cotisationTotal56 = sum(dataRevenuMoyen["Rendement=5.6%_cotisation"])
cotisationTotal56PremiereTranche = sum(dataRevenuMoyen[dataRevenuMoyen["MONTANT_REVENU"]<PASS]["Rendement=5.6%_cotisation"])
cotisationTotal56SecondeTranche = sum(dataRevenuMoyen[dataRevenuMoyen["MONTANT_REVENU"]>=PASS]["Rendement=5.6%_cotisation"])
cotisationTotal56SecondeTranche_plafond = sum(dataRevenuMoyen[(dataRevenuMoyen["MONTANT_REVENU"]>=PASS) & (dataRevenuMoyen["MONTANT_REVENU"]<plafondRevenu)]["Rendement=5.6%_cotisation"])
cotisationTotal56TroisiemeTranche = sum(dataRevenuMoyen[(dataRevenuMoyen["MONTANT_REVENU"]>=plafondRevenu)]["Rendement=5.6%_cotisation"])

dataCotisationsParClasse = dataRevenuMoyen.groupby(by=['classe']).sum()
plt.plot(dataCotisationsParClasse["cotisation_classe"])
plt.plot(dataCotisationsParClasse["Rendement=6.2%_cotisation"])
plt.plot(dataCotisationsParClasse["Rendement=5.6%_cotisation"])
dataCotisationsParClasse.to_csv(adresse + f"dataCotisationsParClasse{inflationRevenus*'_inflation'}.csv")

dataCotisationsParTranche = dataRevenuMoyen.groupby(by=['tranche']).sum()
plt.plot(dataCotisationsParTranche["cotisation_classe"])
plt.plot(dataCotisationsParTranche["Rendement=6.2%_cotisation"])
plt.plot(dataCotisationsParTranche["Rendement=5.6%_cotisation"])
dataCotisationsParTranche.to_csv(adresse + f"dataCotisationsParTranche{inflationRevenus*'_inflation'}.csv")



# INFLATION - Trouver les taux des tranches en fonction de la VA
def trouverTauxSelonRendementPoint(taux):
    absDistances = []
    for index in OutputProjectionPoint.index:
        revenu = OutputProjectionPoint.loc[index, 'Revenu Réel']
        if revenu < PASS:
            absDistances.append(abs(taux[0] * revenu - OutputProjectionPoint.loc[index, f'Cotisation pour VA={round(VA, 1)}']))
        elif revenu >= PASS and revenu < plafondRevenu:
            absDistances.append(abs(taux[0] * PASS + taux[1] * (revenu - PASS) - OutputProjectionPoint.loc[index, f'Cotisation pour VA={round(VA, 1)}']))
    return sum(absDistances)
dicoTauxSelonVA = {}
dicoCotisationSelonVA = {}
dicoTauxRemplacementSelonVA = {}
OutputProjectionPoint = pd.DataFrame({"Revenu Réel": dicoRevenu['Total'], "Nombre de points": [round(dicoCotisationProportionnelle["Rendement=6.2%"][i] / valeurAchatPoint, 0) for i in range(len(dicoCotisationProportionnelle["Rendement=6.2%"]))]})
for inflation in [0.005*i for i in range(11)]:
    VA = (2.63/0.056) * (1 + inflation)
    dicoCotisationSelonVA[VA] = []
    dicoTauxRemplacementSelonVA[VA] = []
    OutputProjectionPoint[f'Cotisation pour VA={round(VA, 1)}'] = [OutputProjectionPoint.loc[i, "Nombre de points"] * VA for i in OutputProjectionPoint.index]
    dicoTauxSelonVA[VA] = list(minimize(trouverTauxSelonRendementPoint, [0.09, 0.22], method='Nelder-Mead').x)
    for index in dataRevenuMoyen.index:
        revenu = dataRevenuMoyen.loc[index, 'MONTANT_REVENU']
        if revenu < PASS:
            dicoCotisationSelonVA[VA].append(dicoTauxSelonVA[VA][0] * revenu)
            dicoTauxRemplacementSelonVA[VA].append(dureeCotisation * rendementDuPoint * dicoTauxSelonVA[VA][0])
        elif revenu >= PASS and revenu < plafondRevenu:
            dicoCotisationSelonVA[VA].append(dicoTauxSelonVA[VA][0] * PASS + dicoTauxSelonVA[VA][1] * (revenu - PASS))
            dicoTauxRemplacementSelonVA[VA].append(dureeCotisation * rendementDuPoint * (dicoTauxSelonVA[VA][0] * PASS + dicoTauxSelonVA[VA][1] * (revenu - PASS)) / revenu)
        else:
            dicoCotisationSelonVA[VA].append(dicoTauxSelonVA[VA][0] * PASS + dicoTauxSelonVA[VA][1] * (plafondRevenu - PASS))
            dicoTauxRemplacementSelonVA[VA].append(dureeCotisation * rendementDuPoint * (dicoTauxSelonVA[VA][0] * PASS + dicoTauxSelonVA[VA][1] * (plafondRevenu - PASS)) / revenu)



# Trouver le taux de remplacement
    VA = 46.96
    rendementDuPoint = 0.056
    taux1 = 0.09
    taux2 = 0.22
    dureeCotisation = 42
    tauxRemplacement = []
    for index in dataRevenuMoyen.index:
        revenu = dataRevenuMoyen.loc[index, 'MONTANT_REVENU']
        if revenu < PASS:
            tauxRemplacement.append(dureeCotisation * rendementDuPoint * taux1)
        elif revenu >= PASS and revenu < plafondRevenu:
            tauxRemplacement.append(dureeCotisation * rendementDuPoint * (taux1 * PASS + taux2 * (revenu - PASS)) / revenu)
        else:
            tauxRemplacement.append(dureeCotisation * rendementDuPoint * (taux1 * PASS + taux2 * (plafondRevenu - PASS)) / revenu)

np.array(tauxRemplacement).mean()





















