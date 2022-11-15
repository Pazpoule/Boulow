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


# Pour les PL ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/Tables/"
data = pd.read_csv(adresse + 'revenus_cotisation_pl.csv', sep=';', encoding='latin-1')
# data.columns = ['NUM_ADHERENT_FONCT', 'ANNEE_COTISATION', 'MONTANT_REVENU', 'Cotisation_appele', 'Cotisation_paye'] # pour les ME
inflation = pd.read_csv(adresse + 'inflation_historique.csv')
dataEvolutionDuPoint = pd.read_excel(adresse + "EvolutionDuPoint.xlsx", index_col=0, sheet_name='EvolutionDuPoint')
data = data.merge(inflation, left_on='ANNEE_COTISATION', right_on='annee', how="left").drop_duplicates()
data.drop('ANNEE_COTISATION', axis=1, inplace=True)
revenu = data[["NUM_ADHERENT_FONCT", "annee", "MONTANT_REVENU"]]
revenu.sort_values(by=["NUM_ADHERENT_FONCT", "annee"], inplace=True, ignore_index=True)
revenu.drop_duplicates(inplace=True)
revenu = revenu[revenu["annee"]>0]
revenu["annee"] = revenu["annee"].astype(int)
revenu = revenu.fillna(0)

# on calcul le taux d'évolution salarial passé
dataEvolutionMoyenneParAnnee = data.groupby(by=["annee"]).median()
dataEvolutionMoyenneParAnnee.reset_index(inplace=True)
dataEvolutionMoyenneParAnnee = dataEvolutionMoyenneParAnnee[["annee", "MONTANT_REVENU"]]
plt.plot(dataEvolutionMoyenneParAnnee['annee'], dataEvolutionMoyenneParAnnee['MONTANT_REVENU'])
tauxEvolutionMoyen = (1+(dataEvolutionMoyenneParAnnee.loc[len(dataEvolutionMoyenneParAnnee)-1, 'MONTANT_REVENU'] - dataEvolutionMoyenneParAnnee.loc[4, 'MONTANT_REVENU'])/dataEvolutionMoyenneParAnnee.loc[4, 'MONTANT_REVENU'])**(1/(len(dataEvolutionMoyenneParAnnee)-4))-1

# # on evalue le revenu moyen récent par adhérents
# dataAnneeRecentes = data[data["annee"].isin([2019, 2020, 2021])]
# tauxRevenuNonPresent = len(dataAnneeRecentes[dataAnneeRecentes["MONTANT_REVENU"].isna()])/len(dataAnneeRecentes) # 3% de revenus non présents
# dataAnneeRecentesHorsNA = dataAnneeRecentes[dataAnneeRecentes["MONTANT_REVENU"]>0]
# dataRevenuMoyen = dataAnneeRecentesHorsNA.groupby(by=["NUM_ADHERENT_FONCT"]).mean()
# dataRevenuMoyen.reset_index(inplace=True)
# dataRevenuMoyen = dataRevenuMoyen.drop(["NUM_ADHERENT", "annee"], axis=1)
# dataRevenuMoyen.sort_values(by=["MONTANT_REVENU"], inplace=True, ignore_index=True)
# plotDistrib({"DistributionRevenuMoyen": dataRevenuMoyen[dataRevenuMoyen["MONTANT_REVENU"]<400000]["MONTANT_REVENU"]})
# # plt.plot(dataRevenuMoyen["MONTANT_REVENU"], dataRevenuMoyen["Cotisation_appele"])

# dataRevenu = revenu[["NUM_ADHERENT_FONCT"]].drop_duplicates()
# dataRevenu.reset_index(inplace=True)
# dataRevenu = pd.DataFrame({"NUM_ADHERENT_FONCT": np.repeat(dataRevenu["NUM_ADHERENT_FONCT"], 81), "annee": [1990+i for i in range(81)]*len(dataRevenu["NUM_ADHERENT_FONCT"])})
# dataRevenu.reset_index(inplace=True)
# dataRevenu = dataRevenu.drop("index", axis=1)

# test = dataRevenu.merge(revenu, on=["NUM_ADHERENT_FONCT", "annee"])
# for index in dataRevenu.index:
#     if (not revenu.loc[index, "MONTANT_REVENU"]>0):
#         if (revenu.loc[index, "NUM_ADHERENT_FONCT"] == revenu.loc[index-1, "NUM_ADHERENT_FONCT"]):
#             dataRevenu.loc[index, "MONTANT_REVENU"] = revenu.loc[index-1, "MONTANT_REVENU"] * (1+tauxEvolutionMoyen)
#         else:
#             revenu.loc[index, "MONTANT_REVENU"] = 0


# On créé la base dataRevenu qui convient
dataRevenu = revenu.join(pd.get_dummies(revenu.annee, prefix='revenu'))
for annee in range(1991, 2022):
    dataRevenu["revenu_"+str(annee)] = dataRevenu["revenu_"+str(annee)]*dataRevenu["MONTANT_REVENU"]
dataRevenu = dataRevenu.groupby(by="NUM_ADHERENT_FONCT").sum()
dataRevenu.reset_index(inplace=True)
dataRevenu = dataRevenu.drop(["annee", "MONTANT_REVENU"], axis=1)
for annee in range(2022, 2071):
    dataRevenu["revenu_"+str(annee)] = 0

# On projette la base dataRevenu avec le taux d'évolution moyen historique
for index in dataRevenu.index:
    print(round(index * 100 / len(dataRevenu), 2), ' %   ')
    for annee in range(1992, 2071):
        if dataRevenu.loc[index, "revenu_"+str(annee)] ==0 and dataRevenu.loc[index, "revenu_"+str(annee-1)]!=0:
            dataRevenu.loc[index, "revenu_" + str(annee)] = dataRevenu.loc[index, "revenu_"+str(annee-1)]*(1+tauxEvolutionMoyen)
dataRevenu.to_csv(adresse + f"projectionRevenu_PL.csv")


plt.plot(dataRevenu.mean()[1:])


dataRevenu = pd.read_csv(adresse + f"projectionRevenu.csv", index_col=0)
def cotisationFonctionRevenu(revenu, taux1=0.09, taux2=0.22, PASS = 41136, plafondRevenu = 123408):
    if revenu < PASS:
        cotisation = (taux1 * revenu)
    elif revenu >= PASS and revenu < plafondRevenu:
        cotisation = (taux1 * PASS + taux2 * (revenu - PASS))
    else:
        cotisation = (taux1 * PASS + taux2 * (plafondRevenu - PASS))
    return cotisation

dataCotisation = dataRevenu.copy()
dataCotisation.columns = ['NUM_ADHERENT_FONCT', 'cotisation_1991', 'cotisation_1992', 'cotisation_1993', 'cotisation_1994', 'cotisation_1995', 'cotisation_1996', 'cotisation_1997', 'cotisation_1998', 'cotisation_1999', 'cotisation_2000', 'cotisation_2001', 'cotisation_2002', 'cotisation_2003', 'cotisation_2004', 'cotisation_2005', 'cotisation_2006', 'cotisation_2007', 'cotisation_2008', 'cotisation_2009', 'cotisation_2010', 'cotisation_2011', 'cotisation_2012', 'cotisation_2013', 'cotisation_2014', 'cotisation_2015', 'cotisation_2016', 'cotisation_2017', 'cotisation_2018', 'cotisation_2019', 'cotisation_2020', 'cotisation_2021', 'cotisation_2022', 'cotisation_2023', 'cotisation_2024', 'cotisation_2025', 'cotisation_2026', 'cotisation_2027', 'cotisation_2028', 'cotisation_2029', 'cotisation_2030', 'cotisation_2031', 'cotisation_2032', 'cotisation_2033', 'cotisation_2034', 'cotisation_2035', 'cotisation_2036', 'cotisation_2037', 'cotisation_2038', 'cotisation_2039', 'cotisation_2040', 'cotisation_2041', 'cotisation_2042', 'cotisation_2043', 'cotisation_2044', 'cotisation_2045', 'cotisation_2046', 'cotisation_2047', 'cotisation_2048', 'cotisation_2049', 'cotisation_2050', 'cotisation_2051', 'cotisation_2052', 'cotisation_2053', 'cotisation_2054', 'cotisation_2055', 'cotisation_2056', 'cotisation_2057', 'cotisation_2058', 'cotisation_2059', 'cotisation_2060', 'cotisation_2061', 'cotisation_2062', 'cotisation_2063', 'cotisation_2064', 'cotisation_2065', 'cotisation_2066', 'cotisation_2067', 'cotisation_2068', 'cotisation_2069', 'cotisation_2070']
for index in dataCotisation.index:
    print(round(index * 100 / len(dataCotisation), 2), ' %   ')
    for annee in range(1992, 2071):
        dataCotisation.loc[index, "cotisation_" + str(annee)] = cotisationFonctionRevenu(dataRevenu.loc[index, "revenu_" + str(annee)])
dataCotisation.to_csv(adresse + f"projectionCotisation_PL.csv")

plt.figure(figsize=[16, 9])
plt.title("Evolution moyenne des cotisations par année")
plt.plot(dataCotisation.sum()[2:])


dataCotisation = pd.read_csv(adresse + f"projectionCotisation_PL.csv", index_col=0)

# On calcul la base dataPrestations dépendant de la base dataCotisation
dataPrestation = dataCotisation.copy()
dataPrestation.columns = ['NUM_ADHERENT_FONCT', 'prestation_1991', 'prestation_1992', 'prestation_1993', 'prestation_1994', 'prestation_1995', 'prestation_1996', 'prestation_1997', 'prestation_1998', 'prestation_1999', 'prestation_2000', 'prestation_2001', 'prestation_2002', 'prestation_2003', 'prestation_2004', 'prestation_2005', 'prestation_2006', 'prestation_2007', 'prestation_2008', 'prestation_2009', 'prestation_2010', 'prestation_2011', 'prestation_2012', 'prestation_2013', 'prestation_2014', 'prestation_2015', 'prestation_2016', 'prestation_2017', 'prestation_2018', 'prestation_2019', 'prestation_2020', 'prestation_2021', 'prestation_2022', 'prestation_2023', 'prestation_2024', 'prestation_2025', 'prestation_2026', 'prestation_2027', 'prestation_2028', 'prestation_2029', 'prestation_2030', 'prestation_2031', 'prestation_2032', 'prestation_2033', 'prestation_2034', 'prestation_2035', 'prestation_2036', 'prestation_2037', 'prestation_2038', 'prestation_2039', 'prestation_2040', 'prestation_2041', 'prestation_2042', 'prestation_2043', 'prestation_2044', 'prestation_2045', 'prestation_2046', 'prestation_2047', 'prestation_2048', 'prestation_2049', 'prestation_2050', 'prestation_2051', 'prestation_2052', 'prestation_2053', 'prestation_2054', 'prestation_2055', 'prestation_2056', 'prestation_2057', 'prestation_2058', 'prestation_2059', 'prestation_2060', 'prestation_2061', 'prestation_2062', 'prestation_2063', 'prestation_2064', 'prestation_2065', 'prestation_2066', 'prestation_2067', 'prestation_2068', 'prestation_2069', 'prestation_2070']
for index in dataPrestation.index:
    print(round(index * 100 / len(dataPrestation), 2), ' %   ')
    for annee in range(1992, 2071):
        dataPrestation.loc[index, "prestation_" + str(annee)] = dataCotisation.loc[index, "cotisation_" + str(annee)] * dataEvolutionDuPoint.loc[annee, "Rendement"]
dataPrestation.to_csv(adresse + f"projectionPrestation_PL.csv")


















data = pd.read_csv(adresse + f"basePL_complete.csv")
data = data.merge(dataRevenuMoyen, on="NUM_ADHERENT_FONCT", how="left")


# ------------------- Choix paramètre -----------------------------------------
rendementDuPoint = 0.0620

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



def cotisationFonctionRevenu(revenu, taux1=0.09, taux2=0.22):
    if revenu < PASS:
        cotisation = (taux1 * revenu)
    elif revenu >= PASS and revenu < plafondRevenu:
        cotisation = (taux1 * PASS + taux2 * (revenu - PASS))
    else:
        cotisation = (taux1 * PASS + taux2 * (plafondRevenu - PASS))
    return cotisation




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



pd.DataFrame(dicoTauxSelonVA).to_csv(adresse + f"tauxSelonVA.csv")
test = pd.DataFrame(dicoCotisationSelonVA)
test.sum()

# tauxRemplacementSelonVA = []
# for inflation in [0.005*i for i in range(2)]:
#     VA = (2.63/0.056) * (1 + inflation)
    # tauxRemplacementSelonVA.append(test[VA].mean())

plotDistrib({"0.5%": pd.DataFrame(dicoTauxRemplacementSelonVA[VA])})

plt.hist(pd.DataFrame(dicoTauxRemplacementSelonVA[VA]))

inflation = 0.005















def encode_data_NN(base, AnneeCalcul=AnneeCalcul):
    print("Execute encode_data_NN ---------------------")
    # base = base[(base["type_ADH"] == "DP") | (base["type_ADH"] == "CER")]
    oneHot_profession = pd.get_dummies(base.profession, prefix='profession')
    oneHot_cp = pd.get_dummies(base.cp, prefix='lieu')
    oneHot_statut = pd.get_dummies(base.Statut_ADH, prefix='statut')
    dicoMoyenne = {}
    dicoSD = {}
    for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "age", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        dicoMoyenne[colonne] = base[colonne].mean()
        dicoSD[colonne] = base[colonne].std()
    base_NN = base[["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "age", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]]
    base_NN["bool_liq_rc"] = 1
    base_NN.loc[base_NN["age_liq_rc"].isna(), "bool_liq_rc"] = 0
    base_NN["bool_rad"] = 1
    base_NN.loc[base_NN["age_rad"].isna(), "bool_rad"] = 0
    base_NN["bool_dc"] = 1
    base_NN.loc[base_NN["age_dc"].isna(), "bool_dc"] = 0
    base_NN["age_liq_rc"] = np.select([base_NN["age_liq_rc"]>0, (base_NN["age_liq_rc"].isna()) & (base_NN["age_dc"].isna()), (base_NN["age_liq_rc"].isna()) & (base_NN["age_dc"]>0)], [base_NN["age_liq_rc"], AnneeCalcul - base_NN["an_nais"], base_NN["age_dc"]])
    base_NN["age_rad"] = np.select([base_NN["age_rad"]>0, (base_NN["age_rad"].isna()) & (base_NN["age_dc"].isna()), (base_NN["age_rad"].isna()) & (base_NN["age_dc"]>0)], [base_NN["age_rad"], AnneeCalcul - base_NN["an_nais"], base_NN["age_dc"]])
    base_NN.loc[base_NN["age_dc"].isna(), "age_dc"] = AnneeCalcul - base_NN["an_nais"]
    base_NN["Sexe"] = 0
    base_NN.loc[base["Homme_Femme"] == "F", "Sexe"] = 1
    base_NN["PL"] = 0
    base_NN.loc[base["PL_ME"] == "PL", "PL"] = 1
    base_NN = base_NN.join(oneHot_profession.join(oneHot_cp.join(oneHot_statut)))
    base_NN["BOOL_VERSEMENT_UNIQUE"] = base["BOOL_VERSEMENT_UNIQUE"].astype('int64')
    for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        base_NN[colonne] = (base_NN[colonne] - dicoMoyenne[colonne]) / dicoSD[colonne]
    base_NN["Validation"] = 0
    base_NN.loc[(base_NN["bool_rad"]==1) & (base_NN["bool_liq_rc"]==1) & (base_NN["bool_dc"]==1) & (base_NN.index % 50 == 0), "Validation"] = 1
    base_NN.reset_index(drop=True, inplace=True)
    print("FIN encode_data_NN ---------------------")
    return base_NN, dicoMoyenne, dicoSD

base_NN, dicoMoyenne, dicoSD = encode_data_NN(data)
# On export ici la base pour la traiter dans le modèle NN
# base_NN.to_csv(adresse + f"Tables/base_NN.csv")

# On import la base résultante du modèle NN
base_NN_projete = pd.read_csv(adresse + 'out.csv')
base_NN_projete.columns = ["age_liq_rc_hat", "age_rad_hat", "age_dc_hat"]
base_NN = data[["NUM_ADHERENT_FONCT", "NUM_ADHERENT", "PointsCotiseParAn"]].join(base_NN.join(base_NN_projete))
# On dénormalise la donnée
for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
    base_NN[colonne] = base_NN[colonne] * dicoSD[colonne] + dicoMoyenne[colonne]
base_NN["age_liq_rc_hat"] = base_NN["age_liq_rc_hat"] * dicoSD["age_liq_rc"] + dicoMoyenne["age_liq_rc"]
base_NN["age_rad_hat"] = base_NN["age_rad_hat"] * dicoSD["age_rad"] + dicoMoyenne["age_rad"]
base_NN["age_dc_hat"] = base_NN["age_dc_hat"] * dicoSD["age_dc"] + dicoMoyenne["age_dc"]
# On applique les seuils de réalisme
base_NN.loc[base_NN["age_liq_rc_hat"]<base_NN["age_1ere_aff"], "age_liq_rc_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_rad_hat"]<base_NN["age_1ere_aff"], "age_rad_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_dc_hat"]<base_NN["age_1ere_aff"], "age_dc_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_liq_rc_hat"]>base_NN["age_dc_hat"], "age_liq_rc_hat"] = base_NN["age_dc_hat"]
base_NN.loc[base_NN["age_rad_hat"]>base_NN["age_dc_hat"], "age_rad_hat"] = base_NN["age_dc_hat"]
base_NN.loc[base_NN["age_liq_rc_hat"]<60, "age_liq_rc_hat"] = 60
for age in ["age_rad", "age_liq_rc", "age_dc", "ageMinRtetraite", "ageTauxPlein", "age_rad_hat", "age_liq_rc_hat", "age_dc_hat"]:
    base_NN[age] = base_NN[age].round(0)

base_NN.to_csv(adresse + f"Tables/base_NN_projete.csv", index=False)


# On reformate la donnée
base_NN = pd.read_csv(adresse + 'Tables/base_NN_projete.csv')
base_NN["PL_ME"] = np.select([(base_NN["PL"] == 1), (base_NN["PL"] == 0)], ["PL", "ME"])
base_NN["Homme_Femme"] = np.select([(base_NN["Sexe"] == 1), (base_NN["Sexe"] == 0)], ["F", "H"])
base_NN["age"] = AnneeCalcul - base_NN["an_nais"]
testVisualisationColonnes = pd.DataFrame(base_NN.columns)
base_NN["profession"] = base_NN[base_NN.columns[19:107]].idxmax(axis=1).str.replace("profession_", "")
base_NN["cp"] = base_NN[base_NN.columns[107:208]].idxmax(axis=1).str.replace("lieu_", "")
base_NN["Statut_ADH"] = base_NN[base_NN.columns[208:213]].idxmax(axis=1).str.replace("statut_", "")
base_NN["type_ADH"] = np.select([(base_NN["age_liq_rc"] < base_NN["age_rad"]), (base_NN["age_liq_rc"] >= base_NN["age_rad"])], ["CER", "DP"])
base_NN.loc[base_NN["Statut_ADH"] == "CER", "type_ADH"] = "CER"
base_NN.drop(base_NN.columns[19:213], inplace=True, axis=1)

base_NN.to_csv(adresse + f"Tables/base_NN_projete.csv", index=False)


# # On créer le dataset de validation et on récupère les identifiants adhérents
# datasetValidation = base_NN[base_NN["Validation"]==1]
# datasetValidation.reset_index(drop=True, inplace=True)
# for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
#     datasetValidation[colonne] = datasetValidation[colonne].round(0)
# data = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
# data["age_liq_rc"] = np.select([data["age_liq_rc"]>0, (data["age_liq_rc"].isna()) & (data["age_dc"].isna()), (data["age_liq_rc"].isna()) & (data["age_dc"]>0)], [data["age_liq_rc"], AnneeCalcul - data["an_nais"], data["age_dc"]])
# data["age_rad"] = np.select([data["age_rad"]>0, (data["age_rad"].isna()) & (data["age_dc"].isna()), (data["age_rad"].isna()) & (data["age_dc"]>0)], [data["age_rad"], AnneeCalcul - data["an_nais"], data["age_dc"]])
# data.loc[data["age_dc"].isna(), "age_dc"] = AnneeCalcul - data["an_nais"]
# for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
#     data[colonne] = data[colonne].round(0)
#
# datasetValidation = datasetValidation.merge(data[["PL_ME", "Homme_Femme", "type_ADH", "profession", "cp", "Statut_ADH", "an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein", "NUM_ADHERENT_FONCT", "NUM_ADHERENT"]], how="inner", on=["PL_ME", "Homme_Femme", "type_ADH", "profession", "cp", "Statut_ADH", "an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]).drop_duplicates(ignore_index=True)
# datasetValidation.to_csv(adresse + f"Tables/datasetValidation.csv", index=False)




























