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
import Passif.Encode_input_data as inputData


adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"
AnneeCalcul = 2021
seuilLourdLeger = 9
filtrePLME = ["ME"]

# On def la baseStat que l'on va utiliser (pour ne pas modifier la base d'export)
baseStat = inputData.base.copy()
baseStat = baseStat[(baseStat["PL_ME"].isin(filtrePLME)) | (baseStat["PL_ME"] == "DD")]
baseStat["an_1ere_aff"] = baseStat["an_nais"] + baseStat["age_1ere_aff"]
baseStat["an_liq_rc"] = baseStat["an_nais"] + baseStat["age_liq_rc"]
baseStat["an_dc"] = baseStat["an_nais"] + baseStat["age_dc"]
baseStat["age_sortie"] = np.array([np.select([baseStat["age_liq_rc"] > 0, baseStat["age_liq_rc"].isna()], [baseStat["age_liq_rc"], math.inf]), np.select([baseStat["age_rad"] > 0, baseStat["age_rad"].isna()], [baseStat["age_rad"], math.inf]), np.select([baseStat["age_dc"] > 0, baseStat["age_dc"].isna()], [baseStat["age_dc"], math.inf])]).min(axis=0)
baseStat["age_derniereCot"] = np.array([baseStat["age_sortie"], baseStat["age"]]).min(axis=0)
baseStat["PointsCotiseParAn"] = baseStat["PTS_RC_CARR"] / np.select([baseStat["age_derniereCot"] - baseStat["age_1ere_aff"]==0, baseStat["age_derniereCot"] - baseStat["age_1ere_aff"]!=0], [1, baseStat["age_derniereCot"] - baseStat["age_1ere_aff"]])
baseStat["Type"] = np.select([
    (baseStat["PointsCotiseParAn"]>=seuilLourdLeger) & (baseStat["Statut_ADH"]=="ACTIF") & (baseStat["age_dc"].isna()),
    (baseStat["PointsCotiseParAn"]<seuilLourdLeger) & (baseStat["Statut_ADH"]=="ACTIF") & (baseStat["age_dc"].isna()),
    (baseStat["Statut_ADH"]=="CER") & (baseStat["age_dc"].isna()),
    (baseStat["type_ADH"].isin(["DP", "CER"])) & (baseStat["Statut_ADH"]=="PRESTATAIRE") & (baseStat["age_dc"].isna()) & ((baseStat["BOOL_VERSEMENT_UNIQUE"]!=1) & (baseStat["PTS_RC_CARR"]>180)),
    (baseStat["type_ADH"].isin(["DP", "CER"])) & (baseStat["Statut_ADH"]=="PRESTATAIRE") & (baseStat["age_dc"].isna()) & ((baseStat["BOOL_VERSEMENT_UNIQUE"]==1) | (baseStat["PTS_RC_CARR"]<=180)),
    (baseStat["type_ADH"]=="DD") & (baseStat["age_dc"].isna()),
    (baseStat["Statut_ADH"]=="RADIE") & (baseStat["age_dc"].isna()),
    (baseStat["an_dc"]>0)],
    ["Lourds", "Légers", "CER", "Retraités", "VFU", "Réversataires", "Radiés", "Décédés"])
baseStat["Groupe_Type"] = np.select([baseStat["Type"].isin(["Lourds", "Légers", "CER"]), baseStat["Type"].isin(["Radiés"]), baseStat["Type"].isin(["Retraités", "VFU", "Réversataires"]), baseStat["Type"].isin(["Décédés"])], ["Cotisants", "Radiés", "Prestataires", "Décédés"])


# Pyramide des Type par génération - En volume
baseStat_tmp = baseStat.copy()
baseStat_tmp.Type = pd.Categorical(baseStat_tmp.Type, categories=["Lourds", "Légers", "Radiés", "CER", "Retraités", "VFU", "Réversataires", "Décédés"], ordered=True)
baseStat_tmp.sort_values(by="Type", inplace=True)
fig = px.histogram(baseStat_tmp, x="an_nais", color="Type")
fig.update_layout(title_text="Distribution des affiliés par génération<br><i>En VOLUME, segmenté par Type", plot_bgcolor='rgb(256,256,256)', bargap=0.2)
fig.show()

# Sunburst des Types - En Volume
baseStat_tmp = pd.DataFrame(baseStat[["Groupe_Type", "Type"]].value_counts())
baseStat_tmp.columns = ["value"]
baseStat_tmp.sort_index(inplace=True)
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.Groupe_Type = pd.Categorical(baseStat_tmp.Groupe_Type, categories=["Cotisants", "Radiés", "Prestataires", "Décédés"], ordered=True)
baseStat_tmp.Type = pd.Categorical(baseStat_tmp.Type, categories=["Lourds", "Légers", "Radiés", "CER", "Retraités", "VFU", "Réversataires", "Décédés"], ordered=True)
fig = px.sunburst(baseStat_tmp, path=['Groupe_Type', 'Type'], values='value')
fig.show()

# Pyramide des Type par génération - En points
baseStat_tmp = baseStat[["Type", "an_nais", "PTS_RC_CARR"]].groupby(["Type", "an_nais"]).sum()
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.Type = pd.Categorical(baseStat_tmp.Type, categories=["Lourds", "Légers", "Radiés", "CER", "Retraités", "VFU", "Réversataires", "Décédés"], ordered=True)
baseStat_tmp.sort_values(by=["Type", "an_nais"], inplace=True)
fig = px.bar(baseStat_tmp, x="an_nais", y="PTS_RC_CARR", color="Type")
fig.update_layout(title_text="Distribution des affiliés par génération<br><i>En POINTS, segmenté par Type", plot_bgcolor='rgb(256,256,256)')
fig.show()

# Sunburst des Types - En Points
baseStat_tmp = baseStat[["Groupe_Type", "Type", "PTS_RC_CARR"]].groupby(["Groupe_Type", "Type"]).sum()
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.Type = pd.Categorical(baseStat_tmp.Type, categories=["Lourds", "Légers", "Radiés", "CER", "Retraités", "VFU", "Réversataires", "Décédés"], ordered=True)
baseStat_tmp.sort_values(by=["Groupe_Type", "Type"], inplace=True)
fig = px.sunburst(baseStat_tmp, path=['Groupe_Type', 'Type'], values='PTS_RC_CARR')
fig.show()

# Piechart des cp
fig = px.pie(baseStat, values='cp', names='cp', title='Répartition géographique')
fig.update_traces(textposition='inside')
fig.show()

# Histogramme des professions
profession = pd.DataFrame(baseStat["profession"].value_counts())
fig = px.bar(profession, x=profession.index, y='profession')
fig.update_layout(title_text="Distribution des professions<br>", plot_bgcolor='rgb(256,256,256)')
fig.show()

# Tables des volumes et des points
baseStat_tmp = pd.DataFrame(baseStat[["Groupe_Type", "Type", "PL_ME"]].value_counts())
baseStat_tmp.columns = ["value"]
baseStat_tmp.sort_index(inplace=True)
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.to_csv(adresse+"Tables/volumeActuels.csv")
baseStat_tmp = baseStat[["Groupe_Type", "Type", "PL_ME", "PointsCotiseParAn"]].groupby(["Type", "PL_ME"]).sum()
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.to_csv(adresse+"Tables/pointsActuelsParAn.csv")
baseStat_tmp = baseStat[["Groupe_Type", "Type", "PL_ME", "PTS_RC_CARR"]].groupby(["Groupe_Type", "Type", "PL_ME"]).sum()
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.to_csv(adresse+"Tables/pointsActuels.csv")
baseStat_tmp = baseStat[["Groupe_Type", "Type", "PL_ME", "POINT_PAYE_RC"]].groupby(["Groupe_Type", "Type", "PL_ME"]).sum()
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.to_csv(adresse+"Tables/pointsActuels_ptspaye.csv")
baseStat_tmp = baseStat[["Groupe_Type", "Type", "PL_ME", "POINT_ACQUI_RC"]].groupby(["Groupe_Type", "Type", "PL_ME"]).sum()
baseStat_tmp.reset_index(inplace=True)
baseStat_tmp.to_csv(adresse+"Tables/pointsActuels_ptsacqui.csv")

# Graph des années de passage par génération
for variable in ["an_nais", "an_1ere_aff", "an_liq_rc", "an_dc"]:
    plt.figure(figsize=[16,9])
    plt.title("Distribution des "+variable)
    distrib = pd.DataFrame(baseStat[variable].value_counts())
    distrib.sort_index(inplace=True)
    plt.plot(distrib[variable])
    distrib = pd.DataFrame(baseStat[(baseStat["Homme_Femme"]=="F")][variable].value_counts())
    distrib.sort_index(inplace=True)
    plt.plot(distrib[variable])
    distrib = pd.DataFrame(baseStat[(baseStat["Homme_Femme"]=="H")][variable].value_counts())
    distrib.sort_index(inplace=True)
    plt.plot(distrib[variable])
plt.figure(figsize=[16, 9])
plt.title("Distributions des années de passage d'un état à un autre")
for variable in ["an_nais", "an_1ere_aff", "an_liq_rc", "an_dc"]:
    distrib = pd.DataFrame(baseStat[variable].value_counts())
    distrib.sort_index(inplace=True)
    plt.plot(distrib[variable])

# Taux interessants
tauxFemme = len(baseStat[(baseStat["Homme_Femme"]=="F") & (baseStat["age_dc"].isna())])/len(baseStat[baseStat["age_dc"].isna()])
print(f"Le taux de femmes en portefeuille (non décédés) est de {round(tauxFemme*100, 0)}%")
tauxPL = len(baseStat[(baseStat["PL_ME"]=="PL") & (baseStat["age_dc"].isna())])/len(baseStat[baseStat["age_dc"].isna()])
print(f"Le taux de PL en portefeuille (non décédés) est de {round(tauxPL*100, 0)}%")

# fig = go.Figure()
# fig.add_trace(go.Violin(x=baseStat['PL_ME'][baseStat['Homme_Femme'] == 'H'],
#                         y=baseStat['an_1ere_aff'][baseStat['Homme_Femme'] == 'H'],
#                         name='Homme',
#                         side='negative',
#                         line_color='lightseagreen')
# )
# fig.add_trace(go.Violin(x=baseStat['PL_ME'][baseStat['Homme_Femme'] == 'F'],
#                         y=baseStat['an_1ere_aff'][baseStat['Homme_Femme'] == 'F'],
#                         name='Femme',
#                         side='positive',
#                         line_color='mediumpurple')
# )
# fig.update_traces(meanline_visible=True, points=False)  # scale violin plot area with total count
# fig.update_layout(title_text="Distribution de l'année d'affiliation<br><i>Segmenté pas sexe et PL/ME", violingap=0, violingroupgap=0, plot_bgcolor='rgb(256,256,256)')
# fig.show()

# baseStat["an_1ere_aff"] = baseStat["an_nais"] + baseStat["age_1ere_aff"]
# fig = go.Figure()
# fig.add_trace(go.Violin(x=baseStat['Statut_ADH'][baseStat['Homme_Femme'] == 'H'],
#                         y=baseStat['an_1ere_aff'][baseStat['Homme_Femme'] == 'H'],
#                         name='Homme',
#                         side='negative',
#                         line_color='lightseagreen')
# )
# fig.add_trace(go.Violin(x=baseStat['Statut_ADH'][baseStat['Homme_Femme'] == 'F'],
#                         y=baseStat['an_1ere_aff'][baseStat['Homme_Femme'] == 'F'],
#                         name='Femme',
#                         side='positive',
#                         line_color='mediumpurple')
# )
# fig.update_traces(meanline_visible=True, points=False)  # scale violin plot area with total count
# fig.update_layout(title_text="Distribution de l'année d'affiliation<br><i>Segmenté pas sexe et PL/ME", violingap=0, violingroupgap=0, plot_bgcolor='rgb(256,256,256)')
# fig.show()









