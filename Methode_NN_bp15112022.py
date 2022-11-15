import matplotlib
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

from Passif.Projection_ages_COUNT import *
from Passif.VolumeAnnuel import *


adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"

def plotDistributionAges(base):
    plt.figure(figsize=[16, 9])
    plt.title("Distribution des ages")
    for age in ["age_1ere_aff", "age_rad", "age_liq_rc", "age_dc", "age_rad_hat", "age_liq_rc_hat", "age_dc_hat", "age_dc_TG", "age_dc_KM"]:
        if age in base.columns:
            distrib = pd.DataFrame(base[age].value_counts())
            distrib.sort_index(inplace=True)
            plt.plot(distrib[age], label=age)
            plt.legend()

def plotDistrib(dico, normalize=True, titre="Distribution"):
    plt.figure(figsize=[16, 9])
    plt.title(titre)
    for index in dico:
        distrib = pd.DataFrame(dico[index].value_counts())
        if normalize: distrib = distrib / distrib.sum()
        distrib.sort_index(inplace=True)
        plt.plot(distrib, label=str(index))
        plt.legend()

def ajoutNouveauAdherents(data, AnneeCalcul=2021, finProj=2070, NombreNouveauxParAns=6000, frequence=0.1):
    # On ajoute des nouveaux entrant en sample et translatant les ages  -  On suppose ici des comportements constant dans le temps (a affiner)
    baseNewSample = pd.DataFrame({})
    for annee in range(AnneeCalcul, finProj):
        tmpNewSample = data[data["type_ADH"]!="DD"].sample(n = int(NombreNouveauxParAns*frequence)) # On exclut les DD (qui n'ont pas de date d'affiliation)
        tmpNewSample['an_nais'] = annee - tmpNewSample['age_1ere_aff']
        baseNewSample = baseNewSample.append(tmpNewSample)
    len(baseNewSample)
    baseNewSample['Type'] = 'NEW'
    data = data.append(baseNewSample)
    data.reset_index(drop=True, inplace=True)
    print('--- Ajout New adh: OK ------------------------------------------------------------------------------------------------')
    return data, baseNewSample

if __name__ == '__main__':

    # Paramètres -----------------------------------------------------------------------------------------------------
    PL_ME = ["PL"]
    debutProj = 2000
    finProj = 2070
    AnneeCalcul = 2021
    NombreNouveauxParAns = 3000*len(PL_ME) # PL ou ME ou les deux
    VA = 45.3
    VS = 2.77
    Rendement = VS / VA
    tauxNuptialite = 0.4    # Nombre d'ayant droit (rattaché à un affilié) / Nombre d'affilié prestataire décédés
    tauxReversion = 0.54
    ageMinReversion = 60
    tauxRecouvrement = 0.95
    coeffVFU = 15
    ajustementTemporellePoints = 1.8

    # Import et traitements --------------------------------------------------------------------------------------------------
    difference_age_DD = pd.read_csv(adresse + 'Tables/difference_age_DD.csv', sep=';', encoding="ISO-8859-1")
    dataEvolutionDuPoint = pd.read_excel(adresse + "Tables/EvolutionDuPoint.xlsx", index_col=0, sheet_name='EvolutionDuPoint')
    data = pd.read_csv(adresse + 'Tables/base_NN_projete.csv') # Exclut les DD, il va falloir les relier séparément
    dataCotisation = pd.read_csv(adresse + f"projectionCotisation.csv", index_col=0)
    data["age"] = AnneeCalcul - data["an_nais"]

    # plotDistributionAges(data)
    # plotDistrib({"original": data[(data["bool_rad"]==1) & (data["bool_dc"]==0)]["age_rad"], "Projeté": data[(data["bool_dc"]==0)]["age_rad_hat"]}, titre="Age de radiation")
    # plotDistrib({"original": data[(data["bool_liq_rc"]==1) & (data["bool_dc"]==0)]["age_liq_rc"], "Projeté": data[(data["bool_dc"]==0)]["age_liq_rc_hat"]}, titre="Age de liquidation")
    # plotDistrib({"original": data[data["bool_dc"]==1]["age_dc"], "Projeté": data["age_dc_hat"]}, titre="Age de décès")
    # test = data[data ["age_dc_hat"]<60]

    data.loc[(data["bool_rad"] == 0) & (data["age"] <= data["age_rad_hat"]), "age_rad"] = data["age_rad_hat"]
    data.loc[(data["bool_liq_rc"] == 0) & (data["age"] <= data["age_liq_rc_hat"]), "age_liq_rc"] = data["age_liq_rc_hat"]
    data.loc[(data["bool_dc"] == 0) & (data["age"] <= data["age_dc_hat"]), "age_dc"] = data["age_dc_hat"]

    # data["age_rad"] = data["age_rad_hat"]
    # data["age_liq_rc"] = data["age_liq_rc_hat"]
    # data["age_dc"] = data["age_dc_hat"]

    data = data[data["PL_ME"].isin(PL_ME)]
    data.reset_index(drop=True, inplace=True)
    # Import la base prest pour récupérer les VFU todo il n'est pas possible de joindre pour récup les VFU

    # data.loc[data["type_ADH"] == "DD", "PTS_RC_CARR"] = data.loc[data["type_ADH"] == "DD", "NBR_POINT_REVERS_RC"] todo gérer les points reversés
    # data.loc[data["type_ADH"] == "DD", "PL_ME"] = "DD"

    # Def des points
    data["age_sortie"] = np.array([data["age_liq_rc"], data["age_rad"], data["age_dc"]]).min(axis=0)
    data["age_derniereCot"] = np.array([data["age_sortie"], data["age"]]).min(axis=0)
    # data["PointsCotiseParAn"] = data["PTS_RC_CARR"] / np.select([data["age_derniereCot"] - data["age_1ere_aff"] == 0, data["age_derniereCot"] - data["age_1ere_aff"] != 0], [1, data["age_derniereCot"] - data["age_1ere_aff"]])
    # data["PointsCotiseParAn"] = data["PointsCotiseParAn"].fillna(0)
    data["PointsAccumule"] = data["PointsCotiseParAn"] * (data["age_derniereCot"] - data["age_1ere_aff"])
    data["PointsAccumule"] = data["PTS_RC_CARR"].fillna(0)

    data.loc[(data["PointsAccumule"] <= 180) & (data["age"] <= data["age_liq_rc"]), "BOOL_VERSEMENT_UNIQUE"] = 1
    data["PointsCotiseParAn"] = ajustementTemporellePoints * data["PointsCotiseParAn"]
    print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

    # Ajout des nouveaux adhérents tous les ans
    data, baseNewSample = ajoutNouveauAdherents(data, AnneeCalcul=AnneeCalcul, finProj=finProj, NombreNouveauxParAns=NombreNouveauxParAns, frequence=1)
    data.to_csv(adresse + f"Tables/basePL_complete.csv", index=False)

    # On projette les volumes par état
    baseProjection = volumeAnnuel(data, frequence=1, debutProj=debutProj, finProj=finProj, tauxRecouvrement=tauxRecouvrement, VA=VA, VS=VS, dataEvolutionDuPoint=dataEvolutionDuPoint, coeffVFU=coeffVFU, tauxReversion=tauxReversion, dataCotisation=dataCotisation, plot=True, lisse=True)
    baseProjection.to_csv(adresse + f"Tables/baseProjection_PL.csv", index=False)






    dataInit = pd.read_csv(adresse + 'Tables/DATA_nettoye.csv')
    dataInit = dataInit[(dataInit["type_ADH"] == "DP") | (dataInit["type_ADH"] == "CER")]
    dataInit = dataInit[dataInit["PL_ME"].isin(PL_ME)]
    dataInit["age"] = AnneeCalcul - dataInit["an_nais"]
    baseProjectionInit = volumeAnnuel(dataInit, frequence=1, debutProj=debutProj, finProj=finProj, tauxRecouvrement=tauxRecouvrement, VA=VA, VS=VS, dataEvolutionDuPoint=dataEvolutionDuPoint, coeffVFU=coeffVFU, tauxReversion=tauxReversion, plot=True)

#
#     # data = dataInit
#     data["an_liq_rc"] = data["an_nais"] + data["age_liq_rc"]
#
#     base = data[(data["an_liq_rc"]<2043) & (data["an_liq_rc"]>2060)]
#     bosse = data[(data["an_liq_rc"]>=2043) & (data["an_liq_rc"]<=2060)]
#
#     plotDistributionAges(data)
#     plotDistributionAges(bosse)
#
# # test = list(bosse.columns)
# # ['Homme_Femme', 'profession', 'cp', 'DECEDE', 'an_nais', 'Statut_ADH', 'PL_ME', 'type_ADH', 'PTS_RC_CARR', 'BOOL_VERSEMENT_UNIQUE', 'age', 'ageMinRtetraite', 'age_1ere_aff', 'age_rad', 'age_liq_rc', 'age_dc', 'PointsCotiseParAn', 'PointsAccumule', 'bool_liq_rc', 'bool_rad', 'bool_dc', 'an_liq_rc']
# #
# # for var in list(bosse.columns):
# #     plotDistrib({"original": data[var], "Bosse": bosse[var]}, titre=var)
#
# for var in list(bosse.columns):
#     plotDistrib({"original, né entre 1960 et 2000": data[(data["an_nais"]>=1960) & (data["an_nais"]<=2000)][var], "Bosse": bosse[(bosse["an_nais"]>=1960) & (bosse["an_nais"]<=2000)][var]}, titre=var)
#
#
#
#     plotDistrib({"Bosse0": bosse[(bosse["an_nais"]>=1960) & (bosse["an_nais"]<=2000) & (bosse["bool_rad"]==1)]["age_rad"], "Bosse": data[(data["bool_rad"]==1)]["age_rad"]}, titre="")
#     plotDistrib({"original, né entre 1960 et 2000": dataInit[(dataInit["an_nais"]>=1960) & (dataInit["an_nais"]<=2000)]["age_rad"], "Bosse": bosse[(bosse["an_nais"]>=1960) & (bosse["an_nais"]<=2000) & (bosse["bool_rad"]==1)]["age_rad"]}, titre="")
#
#
#
# bosse[(bosse["an_nais"]>=1960) & (bosse["an_nais"]<=2000)]
# data[(data["an_nais"]>=1960) & (data["an_nais"]<=2000)]
