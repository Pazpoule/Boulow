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
import plotly.io as pio
pio.renderers.default = "browser"

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"

def lissage(table, force=0.5, plot=False):
    table["taux_lisse"] = table["taux"]
    for i, indice in enumerate(table.index[1:-1]):
        table.loc[indice, "taux_lisse"] = (1 - force) * table.loc[indice, "taux"] + force * (table.loc[table.index[i], "taux"] + table.loc[table.index[i + 2], "taux"]) / 2
    if plot:
        plt.plot(table["taux_lisse"])
    table = table.join(pd.DataFrame(index=[i for i in range(0, 121)]), how='outer').fillna(0)
    return table

def definitionTables_COUNT(data, filtrePLME=["PL", "ME"], plot=False):
    print("Execute tables_COUNT ---------------------")
    data = data[(data["PL_ME"].isin(filtrePLME)) | (data["PL_ME"]=="DD")]
    data.reset_index(drop=True, inplace=True)
    dataCER = data[data["type_ADH"] == "CER"]
    dataDP = data[data["type_ADH"] == "DP"]
    dataDPCER = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]

    # ----------------------------------------------------------------------------------- Définition des différentes tables de passage
    if plot:
        plt.figure(figsize=[16, 9])
        plt.title("Tables de passage")

    if len(dataDP) == 0:
        ageActiftoRad = None
    else:
        ageActiftoRad = pd.DataFrame(dataDP["age_rad"].value_counts())  # On exclut les CER radiés apres liq
        ageActiftoRad.sort_index(inplace=True)
        for indice in ageActiftoRad.index:  # On définit l'assiette par rapport à un age comme étant tous les adh qui sont plus agé mais qui n'ont ni radie avant ni liq ni décédé, et sont bien affilié, on exclu les personne ni adie ni liq ni decede de l'assiette
            ageActiftoRad.loc[indice, "assiette"] = len(dataDP[(dataDP["age"] >= indice) & ((dataDP["age_rad"].isna()) | (dataDP["age_rad"] >= indice)) & ((dataDP["age_liq_rc"].isna()) | (dataDP["age_liq_rc"] >= indice)) & ((dataDP["age_dc"].isna()) | (dataDP["age_dc"] >= indice)) & ((dataDP["age_1ere_aff"].isna()) | (dataDP["age_1ere_aff"] <= indice)) & ((dataDP["age_rad"] > 0) | (dataDP["age_liq_rc"] > 0) | (dataDP["age_dc"] > 0))])
        ageActiftoRad["taux"] = ageActiftoRad["age_rad"] / ageActiftoRad["assiette"]
        ageActiftoRad.sort_index(inplace=True)
        # ageActiftoRad.to_csv(adresse + f"Tables/Loi_ActiftoRad.csv")
        if plot: plt.plot(ageActiftoRad["taux"], label="ageActiftoRad")

    if len(dataCER) == 0:
        ageActiftoCER = None
    else:
        ageActiftoCER = pd.DataFrame(dataCER["age_liq_rc"].value_counts())
        ageActiftoCER.sort_index(inplace=True)
        ageActiftoCER.columns = ["age_liq_rc"]
        for indice in ageActiftoCER.index:
            ageActiftoCER.loc[indice, "assiette"] = len(dataDPCER[(dataDPCER["age"] >= indice) & ((dataDPCER["age_rad"].isna()) | (dataDPCER["age_rad"] >= indice)) & ((dataDPCER["age_liq_rc"].isna()) | (dataDPCER["age_liq_rc"] >= indice)) & ((dataDPCER["age_dc"].isna()) | (dataDPCER["age_dc"] >= indice)) & ((dataDPCER["age_1ere_aff"].isna()) | (dataDPCER["age_1ere_aff"] <= indice)) & ((dataDPCER["age_rad"] > 0) | (dataDPCER["age_liq_rc"] > 0) | (dataDPCER["age_dc"] > 0))])
        ageActiftoCER["taux"] = ageActiftoCER["age_liq_rc"] / ageActiftoCER["assiette"]
        ageActiftoCER.sort_index(inplace=True)
        # ageActiftoCER.to_csv(adresse + f"Tables/Loi_ActiftoCER.csv")
        if plot: plt.plot(ageActiftoCER['taux'], label="ageActiftoCER")

    if len(dataCER) == 0:
        ageCERtoPrest = None
    else:
        ageCERtoPrest = pd.DataFrame(dataCER["age_rad"].value_counts())
        ageCERtoPrest.sort_index(inplace=True)
        for indice in ageCERtoPrest.index:
            ageCERtoPrest.loc[indice, "assiette"] = len(dataCER[(dataCER["age"] >= indice) & ((dataCER["age_rad"].isna()) | (dataCER["age_rad"] >= indice)) & (dataCER["age_liq_rc"] <= indice) & ((dataCER["age_dc"].isna()) | (dataCER["age_dc"] >= indice)) & ((dataCER["age_1ere_aff"].isna()) | (dataCER["age_1ere_aff"] <= indice)) & ((dataCER["age_rad"] > 0) | (dataCER["age_dc"] > 0))])
        ageCERtoPrest["taux"] = ageCERtoPrest["age_rad"] / ageCERtoPrest["assiette"]
        ageCERtoPrest.sort_index(inplace=True)
        # ageCERtoPrest.to_csv(adresse + f"Tables/Loi_CERtoPrest.csv")
        if plot: plt.plot(ageCERtoPrest["taux"], label="ageCERtoPrest")

    if len(dataDP) == 0:
        ageRadtoPrest = None
    else:
        ageRadtoPrest = pd.DataFrame(dataDP.loc[dataDP['age_rad'] > 0, "age_liq_rc"].value_counts())
        ageRadtoPrest.sort_index(inplace=True)
        for indice in ageRadtoPrest.index:
            ageRadtoPrest.loc[indice, "assiette"] = len(dataDP[(dataDP["age"] >= indice) & (dataDP["age_rad"] <= indice) & ((dataDP["age_liq_rc"].isna()) | (dataDP["age_liq_rc"] >= indice)) & ((dataDP["age_dc"].isna()) | (dataDP["age_dc"] >= indice)) & ((dataDP["age_1ere_aff"].isna()) | (dataDP["age_1ere_aff"] <= indice)) & ((dataDP["age_liq_rc"] > 0) | (dataDP["age_dc"] > 0))])
        ageRadtoPrest["taux"] = ageRadtoPrest["age_liq_rc"] / ageRadtoPrest["assiette"]
        ageRadtoPrest.sort_index(inplace=True)
        # ageRadtoPrest.to_csv(adresse + f"Tables/Loi_RadtoPrest.csv")
        if plot: plt.plot(ageRadtoPrest['taux'], label="ageRadtoPrest")

    if len(dataDPCER) == 0 or "age_liq_id" not in dataDPCER.columns:
        ageID = None
    else:
        ageID = pd.DataFrame(dataDPCER["age_liq_id"].value_counts())
        ageID.sort_index(inplace=True)
        for indice in ageID.index:
            ageID.loc[indice, "assiette"] = len(dataDPCER[(dataDPCER["age"] >= indice) & ((dataDPCER["age_liq_id"].isna()) | (dataDPCER["age_liq_id"] >= indice)) & ((dataDPCER["age_dc"].isna()) | (dataDPCER["age_dc"] >= indice)) & ((dataDPCER["age_1ere_aff"].isna()) | (dataDPCER["age_1ere_aff"] <= indice)) & ((dataDPCER["age_liq_id"] > 0) | (dataDPCER["age_dc"] > 0))])
        ageID["taux"] = ageID["age_liq_id"] / ageID["assiette"]
        ageID.sort_index(inplace=True)
        # ageID.to_csv(adresse + f"Tables/Loi_LiqID.csv")
        if plot: plt.plot(ageID['taux'], label="ageID")

    if len(dataDPCER) == 0:
        ageDC = None
    else:
        ageDC = pd.DataFrame(dataDPCER["age_dc"].value_counts())
        ageDC.sort_index(inplace=True)
        ageDC.columns = ["age_dc"]
        for indice in ageDC.index:
            ageDC.loc[indice, "assiette"] = len(dataDPCER[(dataDPCER["age"] >= indice) & (dataDPCER["age_dc"] >= indice) & ((dataDPCER["age_1ere_aff"].isna()) | (dataDPCER["age_1ere_aff"] <= indice)) & (dataDPCER["age_dc"] > 0)])
        ageDC["taux"] = ageDC["age_dc"] / ageDC["assiette"]
        ageDC.sort_index(inplace=True)
        # ageDC.to_csv(adresse + f"Tables/Loi_dc.csv")
        if plot: plt.plot(ageDC['taux'], label="ageDC")

    if len(dataDPCER) == 0:
        ageDC_H = None
    else:
        ageDC_H = pd.DataFrame(dataDPCER[dataDPCER["Homme_Femme"] == "H"]["age_dc"].value_counts())
        ageDC_H.sort_index(inplace=True)
        ageDC_H.columns = ["age_dc"]
        for indice in ageDC_H.index:
            ageDC_H.loc[indice, "assiette"] = len(dataDPCER[dataDPCER["Homme_Femme"] == "H"][(dataDPCER[dataDPCER["Homme_Femme"] == "H"]["age"] >= indice) & (dataDPCER[dataDPCER["Homme_Femme"] == "H"]["age_dc"] >= indice) & ((dataDPCER[dataDPCER["Homme_Femme"] == "H"]["age_1ere_aff"].isna()) | (dataDPCER[dataDPCER["Homme_Femme"] == "H"]["age_1ere_aff"] <= indice)) & (dataDPCER[dataDPCER["Homme_Femme"] == "H"]["age_dc"] > 0)])
        ageDC_H["taux"] = ageDC_H["age_dc"] / ageDC_H["assiette"]
        ageDC_H.sort_index(inplace=True)
        # ageDC_H.to_csv(adresse + f"Tables/Loi_dc.csv")
        if plot: plt.plot(ageDC_H['taux'], label="ageDC_H")

    if len(dataDPCER) == 0:
        ageDC_F = None
    else:
        ageDC_F = pd.DataFrame(dataDPCER[dataDPCER["Homme_Femme"] == "F"]["age_dc"].value_counts())
        ageDC_F.sort_index(inplace=True)
        ageDC_F.columns = ["age_dc"]
        for indice in ageDC_F.index:
            ageDC_F.loc[indice, "assiette"] = len(dataDPCER[dataDPCER["Homme_Femme"] == "F"][(dataDPCER[dataDPCER["Homme_Femme"] == "F"]["age"] >= indice) & (dataDPCER[dataDPCER["Homme_Femme"] == "F"]["age_dc"] >= indice) & ((dataDPCER[dataDPCER["Homme_Femme"] == "F"]["age_1ere_aff"].isna()) | (dataDPCER[dataDPCER["Homme_Femme"] == "F"]["age_1ere_aff"] <= indice)) & (dataDPCER[dataDPCER["Homme_Femme"] == "F"]["age_dc"] > 0)])
        ageDC_F["taux"] = ageDC_F["age_dc"] / ageDC_F["assiette"]
        ageDC_F.sort_index(inplace=True)
        # ageDC_F.to_csv(adresse + f"Tables/Loi_dc.csv")
        if plot: plt.plot(ageDC_F['taux'], label="ageDC_F")

    ageNul = pd.DataFrame({"taux": [0 for i in range(0, 121)], "taux_lisse": [0 for i in range(0, 121)]})

    if plot: plt.legend()

    # -------------------------------------------------------------------------  Lissage et complétion à 0 des ages
    if plot:
        plt.figure(figsize=[16, 9])
        plt.title("Tables de passage lissé")
    if type(ageActiftoRad) == pd.core.frame.DataFrame:
        ageActiftoRad = lissage(ageActiftoRad, plot)
    if type(ageActiftoCER) == pd.core.frame.DataFrame:
        ageActiftoCER = lissage(ageActiftoCER, plot)
    if type(ageRadtoPrest) == pd.core.frame.DataFrame:
        ageRadtoPrest = lissage(ageRadtoPrest, plot)
    if type(ageCERtoPrest) == pd.core.frame.DataFrame:
        ageCERtoPrest = lissage(ageCERtoPrest, plot)
    if type(ageDC) == pd.core.frame.DataFrame:
        ageDC = lissage(ageDC, plot)
    if type(ageDC_H) == pd.core.frame.DataFrame:
        ageDC_H = lissage(ageDC_H, plot)
    if type(ageDC_F) == pd.core.frame.DataFrame:
        ageDC_F = lissage(ageDC_F, plot)

    print('--- FIN tables_COUNT --------------------------------------------------------------------------------------------------')
    return (ageActiftoRad, ageActiftoCER, ageRadtoPrest, ageCERtoPrest, ageDC, ageDC_H, ageDC_F, ageNul)






