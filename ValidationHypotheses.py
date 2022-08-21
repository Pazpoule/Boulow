import numpy as np
import pandas as pd
import datetime
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
import openpyxl
import webbrowser
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import plotly.subplots as sp
import sklearn.metrics as metrics
# import plotly.io as pio
# pio.renderers.default = "browser"
from lifelines import KaplanMeierFitter

from Passif.tables_COUNT import *
from Passif.tables_KM import *
from Passif.Projection_ages_COUNT import *

def plotDistributionAges(base):
    plt.figure(figsize=[16, 9])
    plt.title("Distribution des ages")
    for age in ["age_1ere_aff", "age_1ere_aff", "age_liq_rc", "age_dc", "age_1ere_aff_hat", "age_liq_rc_hat", "age_dc_hat", "age_dc_TG", "age_dc_KM"]:
        if age in base.columns:
            distrib = pd.DataFrame(base[age].value_counts())
            distrib.sort_index(inplace=True)
            plt.plot(distrib[age], label=age)

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"

AnneeCalcul = 2021
TGH05 = pd.read_excel(adresse + "Tables/TableMortaliteTGH05.xlsx", index_col=None, sheet_name='ProbaDC')
TGF05 = pd.read_excel(adresse + "Tables/TableMortaliteTGF05.xlsx", index_col=None, sheet_name='ProbaDC')
difference_age_DD = pd.read_csv(adresse + 'Tables/difference_age_DD.csv', sep=';', encoding="ISO-8859-1")
data = pd.read_csv(adresse + 'Tables/DATA_nettoye.csv')
base_NN = pd.read_csv(adresse + 'Tables/base_NN_projete.csv')
datasetValidation = pd.read_csv(adresse + 'Tables/datasetValidation.csv')
(COUNTActiftoRad, COUNTActiftoCER, COUNTRadtoPrest, COUNTCERtoPrest, COUNTDC, COUNTDC_H, COUNTDC_F, COUNTNul) = definitionTables_COUNT(data=data)
(KMActiftoRad, KMActiftoCER, KMActiftoDC, KMRadtoPrest, KMRadtoDC, KMCERtoPrest, KMCERtoDC, KMPRESTtoDC) = definitionTables_KM(data=data, plot=True)


# --------------------------------------------- Comparaison des tables --------------------------------------------------------
tables_COUNT = [COUNTActiftoRad, COUNTActiftoCER, COUNTRadtoPrest, COUNTCERtoPrest, COUNTDC]
tables_KM = [KMActiftoRad, KMActiftoCER, KMRadtoPrest, KMCERtoPrest, KMPRESTtoDC]
base_NN_COUNT = base_NN.copy()
base_NN_COUNT["age_rad"] = base_NN_COUNT["age_rad_hat"].round(0)
base_NN_COUNT["age_liq_rc"] = base_NN_COUNT["age_liq_rc_hat"].round(0)
base_NN_COUNT["age_dc"] = base_NN_COUNT["age_dc_hat"].round(0)
(NNActiftoRad, NNActiftoCER, NNRadtoPrest, NNCERtoPrest, NNDC, NNDC_H, NNDC_F, NNNul) = definitionTables_COUNT(data=base_NN_COUNT)
tables_NN = [NNActiftoRad, NNActiftoCER, NNRadtoPrest, NNCERtoPrest, NNDC]
for i in range(len(tables_NN)):
    plt.figure(figsize=[16, 9])
    plt.plot(tables_COUNT[i]["taux_lisse"], label="COUNT")
    plt.plot(tables_KM[i], label="KM")
    plt.plot(tables_NN[i]["taux_lisse"], label="NN")
    plt.legend()


# --------------------------------------------- Validation et performance -----------------------------------------------------
baseValidation = data[data["NUM_ADHERENT"].isin(datasetValidation["NUM_ADHERENT"])]
baseValidation["age_rad_reel"] = baseValidation["age_rad"]
baseValidation["age_liq_rc_reel"] = baseValidation["age_liq_rc"]
baseValidation["age_dc_reel"] = baseValidation["age_dc"]
baseValidation["age_rad"] = np.nan
baseValidation["age_liq_rc"] = np.nan
baseValidation["age_dc"] = np.nan
baseValidation["Statut_ADH"] = "ACTIF"
baseValidation["age"] = baseValidation["age_1ere_aff"]
tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC, tauxIntraAnnuel_PresttoDC = definitionTauxIntraAnnuel(data)
base_projete_COUNT_depuis1ereAff, _ = projectionAgesCOUNT(baseValidation, frequence=1, tables=(COUNTActiftoRad, COUNTActiftoCER, COUNTRadtoPrest, COUNTCERtoPrest, COUNTDC, COUNTDC_H, COUNTDC_F, COUNTNul), TGH05=TGH05, TGF05=TGF05, tauxIntraAnnuel_RadtoPrest=tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC=tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_PresttoDC=tauxIntraAnnuel_PresttoDC, tauxIntraAnnuel_CERtoPrest=tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC=tauxIntraAnnuel_CERtoDC, difference_age_DD=difference_age_DD, tauxNuptialite=0, ageMinReversion=60, lissageTaux=True, utiliserTableExp=True)
base_projete_COUNT_depuis1ereAff["age_rad"] = base_projete_COUNT_depuis1ereAff["age_rad"].fillna(base_projete_COUNT_depuis1ereAff["age_dc"])
base_projete_COUNT_depuis1ereAff["age_liq_rc"] = base_projete_COUNT_depuis1ereAff["age_liq_rc"].fillna(base_projete_COUNT_depuis1ereAff["age_dc"])
base_projete_COUNT_depuis1ereAff["age_dc"] = base_projete_COUNT_depuis1ereAff["age_dc"].fillna(base_projete_COUNT_depuis1ereAff["age_dc"])

loss_rad_COUNT_depuis1ereAff = abs(base_projete_COUNT_depuis1ereAff["age_rad_reel"] - base_projete_COUNT_depuis1ereAff["age_rad"]).mean()
loss_liq_rc_COUNT_depuis1ereAff = abs(base_projete_COUNT_depuis1ereAff["age_liq_rc_reel"] - base_projete_COUNT_depuis1ereAff["age_liq_rc"]).mean()
loss_dc_COUNT_depuis1ereAff = abs(base_projete_COUNT_depuis1ereAff["age_dc_reel"] - base_projete_COUNT_depuis1ereAff["age_dc"]).mean()

loss_rad_NN = abs(datasetValidation["age_rad"]-datasetValidation["age_rad_hat"]).mean()
loss_liq_rc_NN = abs(datasetValidation["age_liq_rc"]-datasetValidation["age_liq_rc_hat"]).mean()
loss_dc_NN = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_hat"]).mean()

ageMoyenBaseActif = (AnneeCalcul - data[(data["age_dc"].isna()) & (data["age_liq_rc"].isna()) & (data["age_rad"].isna())]["an_nais"]).mean()
ageMoyenBaseRadie = (AnneeCalcul - data[(data["age_dc"].isna()) & (data["age_liq_rc"].isna()) & (data["age_rad"]>0)]["an_nais"]).mean()
ageMoyenBase = (AnneeCalcul - data[data["age_dc"].isna()]["an_nais"]).mean()
# Décès par table de mortalité
for adh in datasetValidation.index:
    print(round(adh*100/len(datasetValidation), 2), ' %         Boucle dc TG ageMoyen')
    tableMortalite = TGF05 if datasetValidation.loc[adh, 'Homme_Femme'] == "F" else TGH05
    for age in range(int(ageMoyenBase), 120):
        if random.random() <= tableMortalite.loc[age, datasetValidation.loc[adh, 'an_nais']]:
            datasetValidation.loc[adh, "age_dc_TG_ageMoyen"] = age
            break
loss_dc_TG_ageMoyen = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_TG_ageMoyen"]).mean()

for adh in datasetValidation.index:
    print(round(adh*100/len(datasetValidation), 2), ' %         Boucle dc TG 5ans')
    tableMortalite = TGF05 if datasetValidation.loc[adh, 'Homme_Femme'] == "F" else TGH05
    for age in range(int(datasetValidation.loc[adh, 'age_dc']-5), 120):
        if random.random() <= tableMortalite.loc[age, datasetValidation.loc[adh, 'an_nais']]:
            datasetValidation.loc[adh, "age_dc_TG_5ans"] = age
            break
loss_dc_TG_5ans = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_TG_5ans"]).mean()

for adh in datasetValidation.index:
    print(round(adh*100/len(datasetValidation), 2), ' %         Boucle dc TG 1ereAff')
    tableMortalite = TGF05 if datasetValidation.loc[adh, 'Homme_Femme'] == "F" else TGH05
    for age in range(int(datasetValidation.loc[adh, 'age_1ere_aff']), 120):
        if random.random() <= tableMortalite.loc[age, datasetValidation.loc[adh, 'an_nais']]:
            datasetValidation.loc[adh, "age_dc_TG_1ereAff"] = age
            break
loss_dc_TG_1ereAff = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_TG_1ereAff"]).mean()

datasetValidation["age_rad_KM_ageMoyen"] = 0
datasetValidation["age_liq_rc_KM_ageMoyen"] = 0
datasetValidation["age_dc_KM_ageMoyen"] = 0
for adh in datasetValidation.index:
    print(round(adh*100/len(datasetValidation), 2), ' %         Boucle dc KM age Moyen')
    for age in range(int(ageMoyenBaseActif), 120):
        if random.random() <= (0 if (age - datasetValidation.loc[adh, 'age_1ere_aff']) not in KMActiftoRad.index else KMActiftoRad.loc[age - datasetValidation.loc[adh, 'age_1ere_aff'], "KM_estimate"]):
            datasetValidation.loc[adh, "age_rad_KM_ageMoyen"] = age
            break
    for age in range(int(ageMoyenBaseRadie), 120):
        if random.random() <= (0 if (age - datasetValidation.loc[adh, 'age_rad']) not in KMRadtoPrest.index else KMRadtoPrest.loc[age - datasetValidation.loc[adh, 'age_rad'], "KM_estimate"]):
            datasetValidation.loc[adh, "age_liq_rc_KM_ageMoyen"] = age
            break
    for age in range(int(ageMoyenBase), 120):
        if random.random() <= (0 if (age - max(datasetValidation.loc[adh, 'age_liq_rc'], datasetValidation.loc[adh, 'age_rad'])) not in KMPRESTtoDC.index else KMPRESTtoDC.loc[age - max(datasetValidation.loc[adh, 'age_liq_rc'], datasetValidation.loc[adh, 'age_rad']), "KM_estimate"]):
            datasetValidation.loc[adh, "age_dc_KM_ageMoyen"] = age
            break
loss_rad_KM_ageMoyen = abs(datasetValidation["age_rad"]-datasetValidation["age_rad_KM_ageMoyen"]).mean()
loss_liq_rc_KM_ageMoyen = abs(datasetValidation["age_liq_rc"]-datasetValidation["age_liq_rc_KM_ageMoyen"]).mean()
loss_dc_KM_ageMoyen = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_KM_ageMoyen"]).mean()

datasetValidation["age_rad_KM_5ans"] = 0
datasetValidation["age_liq_rc_KM_5ans"] = 0
datasetValidation["age_dc_KM_5ans"] = 0
for adh in datasetValidation.index:
    print(round(adh*100/len(datasetValidation), 2), ' %         Boucle dc KM 5 ans')
    for age in range(int(datasetValidation.loc[adh, 'age_rad']-5), 120):
        if random.random() <= (0 if (age - datasetValidation.loc[adh, 'age_1ere_aff']) not in KMActiftoRad.index else KMActiftoRad.loc[age - datasetValidation.loc[adh, 'age_1ere_aff'], "KM_estimate"]):
            datasetValidation.loc[adh, "age_rad_KM_5ans"] = age
            break
    for age in range(int(datasetValidation.loc[adh, 'age_liq_rc'] - 5), 120):
        if random.random() <= (0 if (age - datasetValidation.loc[adh, 'age_rad']) not in KMRadtoPrest.index else KMRadtoPrest.loc[age - datasetValidation.loc[adh, 'age_rad'], "KM_estimate"]):
            datasetValidation.loc[adh, "age_liq_rc_KM_5ans"] = age
            break
    for age in range(int(datasetValidation.loc[adh, 'age_dc'] - 5), 120):
        if random.random() <= (0 if (age - max(datasetValidation.loc[adh, 'age_liq_rc'], datasetValidation.loc[adh, 'age_rad'])) not in KMPRESTtoDC.index else KMPRESTtoDC.loc[age - max(datasetValidation.loc[adh, 'age_liq_rc'], datasetValidation.loc[adh, 'age_rad']), "KM_estimate"]):
            datasetValidation.loc[adh, "age_dc_KM_5ans"] = age
            break
loss_rad_KM_5ans = abs(datasetValidation["age_rad"]-datasetValidation["age_rad_KM_5ans"]).mean()
loss_liq_rc_KM_5ans = abs(datasetValidation["age_liq_rc"]-datasetValidation["age_liq_rc_KM_5ans"]).mean()
loss_dc_KM_5ans = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_KM_5ans"]).mean()


datasetValidation["age_rad_KM_1ereAff"] = 0
datasetValidation["age_liq_rc_KM_1ereAff"] = 0
datasetValidation["age_dc_KM_1ereAff"] = 0
for adh in datasetValidation.index:
    print(round(adh*100/len(datasetValidation), 2), ' %         Boucle dc KM 1ere aff')
    for age in range(int(datasetValidation.loc[adh, 'age_1ere_aff']), 120):
        if random.random() <= (0 if (age - datasetValidation.loc[adh, 'age_1ere_aff']) not in KMActiftoRad.index else KMActiftoRad.loc[age - datasetValidation.loc[adh, 'age_1ere_aff'], "KM_estimate"]):
            datasetValidation.loc[adh, "age_rad_KM_1ereAff"] = age
            break
    for age in range(int(datasetValidation.loc[adh, 'age_1ere_aff']), 120):
        if random.random() <= (0 if (age - datasetValidation.loc[adh, 'age_rad']) not in KMRadtoPrest.index else KMRadtoPrest.loc[age - datasetValidation.loc[adh, 'age_rad'], "KM_estimate"]):
            datasetValidation.loc[adh, "age_liq_rc_KM_1ereAff"] = age
            break
    for age in range(int(datasetValidation.loc[adh, 'age_1ere_aff']), 120):
        if random.random() <= (0 if (age - max(datasetValidation.loc[adh, 'age_liq_rc'], datasetValidation.loc[adh, 'age_rad'])) not in KMPRESTtoDC.index else KMPRESTtoDC.loc[age - max(datasetValidation.loc[adh, 'age_liq_rc'], datasetValidation.loc[adh, 'age_rad']), "KM_estimate"]):
            datasetValidation.loc[adh, "age_dc_KM_1ereAff"] = age
            break
loss_rad_KM_1ereAff = abs(datasetValidation["age_rad"]-datasetValidation["age_rad_KM_1ereAff"]).mean()
loss_liq_rc_KM_1ereAff = abs(datasetValidation["age_liq_rc"]-datasetValidation["age_liq_rc_KM_1ereAff"]).mean()
loss_dc_KM_1ereAff = abs(datasetValidation["age_dc"]-datasetValidation["age_dc_KM_1ereAff"]).mean()

















