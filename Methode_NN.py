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
    for age in ["age_1ere_aff", "age_1ere_aff", "age_liq_rc", "age_dc", "age_1ere_aff_hat", "age_liq_rc_hat", "age_dc_hat", "age_dc_TG", "age_dc_KM"]:
        if age in base.columns:
            distrib = pd.DataFrame(base[age].value_counts())
            distrib.sort_index(inplace=True)
            plt.plot(distrib[age], label=age)

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

    PL_ME = ["PL", "ME"]
    debutProj = 2000
    finProj = 2070
    AnneeCalcul = 2021
    NombreNouveauxParAns = 3000 # PL ou ME ou les deux
    VA = 46.96428571428571
    VS = 2.63
    Rendement = VS / VA
    tauxNuptialite = 0.4    # Nombre d'ayant droit (rattaché à un affilié) / Nombre d'affilié prestataire décédés
    tauxReversion = 0.54
    ageMinReversion = 60
    tauxRecouvrement = 0.95
    coeffVFU = 15
    ajustementTemporellePoints = 1.8


    difference_age_DD = pd.read_csv(adresse + 'Tables/difference_age_DD.csv', sep=';', encoding="ISO-8859-1")
    dataEvolutionDuPoint = pd.read_excel(adresse + "Tables/EvolutionDuPoint.xlsx", index_col=0, sheet_name='EvolutionDuPoint')
    data = pd.read_csv(adresse + 'Tables/base_NN_projete.csv')
    data = data[(data["PL_ME"].isin(PL_ME)) | (data["PL_ME"] == "DD")]
    data.reset_index(drop=True, inplace=True)
    data["PointsCotiseParAn"] = ajustementTemporellePoints * data["PointsCotiseParAn"]
    # baseProjection_preProjection = volumeAnnuel(data.sample(frac=frequence), frequence=frequence, debutProj=debutProj, finProj=AnneeCalcul, tauxRecouvrement=tauxRecouvrement, VA=VA, VS=VS, dataEvolutionDuPoint=dataEvolutionDuPoint, coeffVFU=coeffVFU, tauxReversion=tauxReversion, plot=True)

    print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

    # On projette les ages
    tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC, tauxIntraAnnuel_PresttoDC = definitionTauxIntraAnnuel(data)
    tables_COUNT = definitionTables_COUNT(data, PL_ME, plot=True)
    data, baseNouveauDD = projectionAgesCOUNT(data, frequence, tables_COUNT, TGH05, TGF05, tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_PresttoDC, tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC, difference_age_DD, tauxNuptialite, ageMinReversion=ageMinReversion, lissageTaux=True, utiliserTableExp=False)
    # baseProjection_postProjection = volumeAnnuel(data, frequence=frequence, debutProj=debutProj, finProj=AnneeCalcul, tauxRecouvrement=tauxRecouvrement, VA=VA, VS=VS, dataEvolutionDuPoint=dataEvolutionDuPoint, coeffVFU=coeffVFU, tauxReversion=tauxReversion, plot=True)

    # Ajout des nouveaux adhérents tous les ans
    data, baseNewSample = ajoutNouveauAdherents(data, AnneeCalcul=AnneeCalcul, finProj=finProj, NombreNouveauxParAns=NombreNouveauxParAns, frequence=frequence)
    # baseProjection_postNouveau = volumeAnnuel(data, frequence=frequence, debutProj=debutProj, finProj=AnneeCalcul, tauxRecouvrement=tauxRecouvrement, VA=VA, VS=VS, dataEvolutionDuPoint=dataEvolutionDuPoint, coeffVFU=coeffVFU, tauxReversion=tauxReversion, plot=True)

    # On met à jour les points Accumulé et les VFU
    data["age_sortie"] = np.array([np.select([data["age_liq_rc"] > 0, data["age_liq_rc"].isna()], [data["age_liq_rc"], math.inf]), np.select([data["age_rad"] > 0, data["age_rad"].isna()], [data["age_rad"], math.inf]), np.select([data["age_dc"] > 0, data["age_dc"].isna()], [data["age_dc"], math.inf])]).min(axis=0)
    data["age_derniereCot"] = np.array([data["age_sortie"], data["age"]]).min(axis=0)
    data["PointsAccumule"] = data["PointsCotiseParAn"] * (data["age_derniereCot"] - data["age_1ere_aff"])
    data["PointsCotiseParAn"] = data["PointsCotiseParAn"].fillna(0)
    data["PointsAccumule"] = data["PointsAccumule"].fillna(0)
    data.loc[data["PointsAccumule"] <= 180, "BOOL_VERSEMENT_UNIQUE"] = 1

    # On projette les volumes par état
    baseProjection = volumeAnnuel(data, frequence=frequence, debutProj=debutProj, finProj=finProj, tauxRecouvrement=tauxRecouvrement, VA=VA, VS=VS, dataEvolutionDuPoint=dataEvolutionDuPoint, coeffVFU=coeffVFU, tauxReversion=tauxReversion, plot=True)

    baseProjection.to_csv(adresse + f"Tables/baseProjection.csv", index=False)


    # import Passif.bp.Encode_input_data_bp30032022 as inputData
    # data = inputData.base
    # data = pd.read_csv(adresse + 'Tables/DATA_nettoye.csv')
    # data = data[(data["PL_ME"].isin(PL_ME)) | (data["PL_ME"]=="DD")]
    # data.reset_index(drop=True, inplace=True)
    # testProjection = volumeAnnuel(data, 1, 2000, 2070, 1, VA, VS, dataEvolutionDuPoint, coeffVFU, plot=True)

    # baseNewSample["an_1ere_aff"] = baseNewSample["an_nais"] + baseNewSample["age_1ere_aff"]
    # distrib = pd.DataFrame(baseNewSample[(baseNewSample["Homme_Femme"]=="H")]["an_1ere_aff"].value_counts())
    # distrib.sort_index(inplace=True)
    # distrib.mean()











