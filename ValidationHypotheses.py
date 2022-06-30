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
COUNTActiftoRad, COUNTActiftoCER, COUNTRadtoPrest, COUNTCERtoPrest, COUNTDC, COUNTDC_H, COUNTDC_F, COUNTNul = definitionTables_COUNT(data=data)
KMActiftoRad, KMActiftoCER, KMActiftoDC, KMRadtoPrest, KMRadtoDC, KMCERtoPrest, KMCERtoDC, KMPRESTtoDC = definitionTables_KM(data=data)


# --------------------------------------------- Comparaison des tables --------------------------------------------------------
tables_COUNT = [COUNTActiftoRad, COUNTActiftoCER, COUNTRadtoPrest, COUNTCERtoPrest]
tables_KM = [KMActiftoRad, KMActiftoCER, KMRadtoPrest, KMCERtoPrest]
base_NN_COUNT = base_NN.copy()
base_NN_COUNT["age_rad"] = base_NN_COUNT["age_rad_hat"]
base_NN_COUNT["age_liq_rc"] = base_NN_COUNT["age_liq_rc_hat"]
base_NN_COUNT["age_dc"] = base_NN_COUNT["age_dc_hat"]
NNActiftoRad, NNActiftoCER, NNRadtoPrest, NNCERtoPrest, NNDC, NNDC_H, NNDC_F, NNNul = definitionTables_COUNT(data=base_NN_COUNT)
tables_NN = [NNActiftoRad, NNActiftoCER, NNRadtoPrest, NNCERtoPrest]
for i in range(len(tables_NN)):
    plt.figure(figsize=[16, 9])
    plt.plot(tables_COUNT[i]["taux_lisse"], label="COUNT")
    plt.plot(tables_KM[i], label="KM")
    plt.plot(tables_NN[i]["taux_lisse"], label="NN")


# --------------------------------------------- Validation et performance -----------------------------------------------------
baseValidation = data[data["NUM_ADHERENT"].isin(datasetValidation["NUM_ADHERENT"])]
tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC, tauxIntraAnnuel_PresttoDC = definitionTauxIntraAnnuel(data)
base_projete_COUNT = projectionAgesCOUNT(baseValidation, frequence=1, TGH05=TGH05, TGF05=TGF05, tauxIntraAnnuel_RadtoPrest=tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC=tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_PresttoDC=tauxIntraAnnuel_PresttoDC, tauxIntraAnnuel_CERtoPrest=tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC=tauxIntraAnnuel_CERtoDC, difference_age_DD=difference_age_DD, tauxNuptialite=0.4, lissageTaux=True, utiliserTableExp=True)

datasetValidation = base_NN[base_NN["Validation"]==1]
datasetValidation.reset_index(drop=True, inplace=True)


ageMoyenBase = (AnneeCalcul - base_NN[base_NN["bool_dc"]==0]["an_nais"]).mean()

# Décès par table de mortalité
for adh in base_NN[base_NN["Validation"]==1].index:
    print(round(adh*100/len(base_NN), 2), ' %         Boucle décès par table de mortalité')
    tableMortalite = TGF05 if base_NN.loc[adh, 'Sexe'] == 1 else TGH05
    for age in range(int(ageMoyenBase), 120):
        if random.random() <= tableMortalite.loc[age, base_NN.loc[adh, 'an_nais']]:
            base_NN.loc[adh, "age_dc_TG"] = age
            break

# for adh in base_NN[base_NN["Validation"]==1].index:
#     print(round(adh*100/len(base_NN), 2), ' %         Boucle décès par table de mortalité')
#     tableMortalite = TGF05 if base_NN.loc[adh, 'Sexe'] == 1 else TGH05
#     for age in range(int(base_NN.loc[adh, 'age_dc']-5), 120):
#         if random.random() <= tableMortalite.loc[age, base_NN.loc[adh, 'an_nais']]:
#             base_NN.loc[adh, "age_dc_TG"] = age
#             break


abs(datasetValidation["age_rad"]-datasetValidation["age_rad_hat"]).mean()
abs(datasetValidation["age_liq_rc"]-datasetValidation["age_liq_rc_hat"]).mean()
abs(datasetValidation["age_dc"]-datasetValidation["age_dc_hat"]).mean()
abs(datasetValidation["age_dc"]-datasetValidation["age_dc_TG"]).mean()



dataCER = base[base["type_ADH"] == "CER"]
dataDP = base[base["type_ADH"] == "DP"]
dataDPCER = base[(base["type_ADH"] == "DP") | (base["type_ADH"] == "CER")]

# Kaplan-Meier
# Depuis l'état de cotisant :
dataDP["age_sortie"] = np.array([np.select([dataDP["age_liq_rc"] > 0, dataDP["age_liq_rc"].isna()], [dataDP["age_liq_rc"], math.inf]), np.select([dataDP["age_rad"] > 0, dataDP["age_rad"].isna()], [dataDP["age_rad"], math.inf]), np.select([dataDP["age_dc"] > 0, dataDP["age_dc"].isna()], [dataDP["age_dc"], math.inf])]).min(axis=0)
dataDP["age_derniereCot"] = np.array([dataDP["age_sortie"], dataDP["age"]]).min(axis=0)
KM_duration_actif = pd.DataFrame(dataDP["age_derniereCot"]-dataDP["age_1ere_aff"])
# actif to rad
KM_observation_actiftoRad = pd.DataFrame(np.select([(dataDP["age_rad"] <= dataDP["age_liq_rc"]) | (dataDP["age_rad"] <= dataDP["age_dc"]) | ((dataDP["age_rad"]>0) & (dataDP["age_liq_rc"].isna()) & (dataDP["age_dc"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_actif, event_observed=KM_observation_actiftoRad)
kmf.survival_function_.plot(title="ACTIF to RAD")
tableActiftoRad = 1 - kmf.survival_function_
# actif to CER
KM_observation_actiftoCER = pd.DataFrame(np.select([(dataDP["age_liq_rc"] <= dataDP["age_rad"]) | (dataDP["age_liq_rc"] <= dataDP["age_dc"]) | ((dataDP["age_liq_rc"]>0) & (dataDP["age_rad"].isna()) & (dataDP["age_dc"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_actif, event_observed=KM_observation_actiftoCER)
kmf.survival_function_.plot(title="ACTIF to CER")
tableActiftoCER = 1 - kmf.survival_function_
# actif to DC
KM_observation_actiftoDC = pd.DataFrame(np.select([((dataDP["age_dc"]>0) & (dataDP["age_rad"].isna()) & (dataDP["age_liq_rc"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_actif, event_observed=KM_observation_actiftoDC)
kmf.survival_function_.plot(title="ACTIF to DC")
tableActiftoDC = 1 - kmf.survival_function_
# Depuis l'état de radié :
dataDP_radie = dataDP[(dataDP["age_rad"]<dataDP["age_liq_rc"]) | ((dataDP["age_rad"]>0) & (dataDP["age_liq_rc"].isna()))]
dataDP_radie["age_sortie"] = np.array([np.select([dataDP_radie["age_liq_rc"] > 0, dataDP_radie["age_liq_rc"].isna()], [dataDP_radie["age_liq_rc"], math.inf]), np.select([dataDP_radie["age_dc"] > 0, dataDP_radie["age_dc"].isna()], [dataDP_radie["age_dc"], math.inf]), dataDP_radie["age"]]).min(axis=0)
KM_duration_radie = pd.DataFrame(dataDP_radie["age_sortie"]-dataDP_radie["age_rad"])
# rad to prest
KM_observation_radtoPrest = pd.DataFrame(np.select([(dataDP_radie["age_liq_rc"] <= dataDP_radie["age_dc"]) | ((dataDP_radie["age_liq_rc"]>0) & (dataDP_radie["age_dc"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_radie, event_observed=KM_observation_radtoPrest)
kmf.survival_function_.plot(title="RAD to PREST")
tableRadtoPrest = 1 - kmf.survival_function_
# rad to DC
KM_observation_radtoDC = pd.DataFrame(np.select([((dataDP_radie["age_dc"]>0) & (dataDP_radie["age_liq_rc"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_radie, event_observed=KM_observation_radtoDC)
kmf.survival_function_.plot(title="RAD to DC")
tableRadtoDC = 1 - kmf.survival_function_
# Depuis l'état de CER :
dataDP_CER = dataDPCER[(dataDPCER["age_liq_rc"] < dataDPCER["age_rad"]) | ((dataDPCER["age_liq_rc"]>0) & (dataDPCER["age_rad"].isna()))]
dataDP_CER["age_sortie"] = np.array([np.select([dataDP_CER["age_rad"] > 0, dataDP_CER["age_rad"].isna()], [dataDP_CER["age_rad"], math.inf]), np.select([dataDP_CER["age_dc"] > 0, dataDP_CER["age_dc"].isna()], [dataDP_CER["age_dc"], math.inf]), dataDP_CER["age"]]).min(axis=0)
KM_duration_CER = pd.DataFrame(dataDP_CER["age_sortie"]-dataDP_CER["age_liq_rc"])
# CER to prest
KM_observation_CERtoPrest = pd.DataFrame(np.select([(dataDP_CER["age_rad"] <= dataDP_CER["age_dc"]) | ((dataDP_CER["age_rad"]>0) & (dataDP_CER["age_dc"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_CER, event_observed=KM_observation_CERtoPrest)
kmf.survival_function_.plot(title="CER to PREST")
tableCERtoPrest = 1 - kmf.survival_function_
# CER to DC
KM_observation_CERtoDC = pd.DataFrame(np.select([((dataDP_CER["age_dc"]>0) & (dataDP_CER["age_rad"].isna()))], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_CER, event_observed=KM_observation_CERtoDC)
kmf.survival_function_.plot(title="CER to DC")
tableCERtoDC = 1 - kmf.survival_function_
# Depuis l'état de PREST :
dataDP_PREST = dataDPCER[(dataDPCER["age_liq_rc"] > 0) & (dataDPCER["age_rad"] > 0)]
dataDP_PREST["age_sortie"] = np.array([np.select([dataDP_PREST["age_dc"] > 0, dataDP_PREST["age_dc"].isna()], [dataDP_PREST["age_dc"], math.inf]), dataDP_PREST["age"]]).min(axis=0)
KM_duration_PREST = pd.DataFrame(dataDP_PREST["age_sortie"]-np.array([dataDP_PREST["age_liq_rc"], dataDP_PREST["age_rad"]]).max(axis=0))
# PREST to DC
KM_observation_PREST = pd.DataFrame(np.select([dataDP_PREST["age_dc"] > 0], [1]))
kmf = KaplanMeierFitter()
kmf.fit(KM_duration_PREST, event_observed=KM_observation_PREST)
kmf.survival_function_.plot(title="PREST to DC")
tablePRESTtoDC = 1 - kmf.survival_function_

base_NN["age_dc_KM"] = 0
for adh in base_NN[base_NN["Validation"]==1].index:
    print(round(adh*100/len(base_NN), 2), ' %         Boucle décès par table de mortalité')
    for age in range(int(base_NN.loc[adh, 'age_dc']-5), 120):
        if random.random() <= (0 if (age - max(base_NN.loc[adh, 'age_liq_rc'], base_NN.loc[adh, 'age_rad'])) not in tablePRESTtoDC.index else tablePRESTtoDC.loc[age - max(base_NN.loc[adh, 'age_liq_rc'], base_NN.loc[adh, 'age_rad']), "KM_estimate"]):
            base_NN.loc[adh, "age_dc_KM"] = age
            break


# todo proprifier la donnée (ajouter les seuil)











#todo à mettre avec la validation des hypotheses
# On compare la table de mortalité nationnal pour l'année 1975 (génération moyenne)
plt.figure(figsize=[16, 9])
plt.title("Comparaison des tables de mortalité")
data[data["age_dc"].isna()]["an_nais"].mean()
tableMortalite = pd.DataFrame({"taux": TGH05[1975], "taux_lisse": TGH05[1975]})
plt.plot(tableMortalite["taux_lisse"])
plt.plot(ageDC_H["taux_lisse"])





# # Ecart de longévité -----------------------------------------------
# dataDC["ageDCReel"] = dataDC["an_dc"] - dataDC["an_nais"]
# dataDC = dataDC[dataDC["ageDCReel"]>0]
# dataDC.reset_index(drop=True, inplace=True)
# for adh in range(len(dataDC)):
#     print(round(adh*100/len(dataDC), 2), ' %         Boucle projection des décès déjà réalisés')
#     tableMortalite = TGF05 if dataDC.loc[adh, 'Homme_Femme'] == 'F' else TGH05
#     for age in range(120):
#         if dataDC.loc[adh, 'an_nais'] <= 2005:         # On fixe a 120ans (on ne tue pas) les adh qui sont né trop tard (apres 2005) (ils ont 16ans en 2021)
#             if random.random() <= tableMortalite.loc[age, dataDC.loc[adh, 'an_nais']]:
#                 dataDC.loc[adh, 'ageDCProjete'] = age
#                 break
# dataDC["ecartLongevite"] = dataDC["ageDCReel"] - dataDC['ageDCProjete']
# plt.plot(dataDC["ageDCReel"])
# plt.plot(dataDC['ageDCProjete'])
# plt.plot(dataDC["ecartLongevite"])
# dataDC["ageDCReel"].mean()
# dataDC['ageDCProjete'].mean()
# dataDC["ecartLongevite"].mean()



# --------------------------------------------- Comportement annuel projeté ---------------------------------------------------



