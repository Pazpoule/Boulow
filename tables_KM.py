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
from lifelines import KaplanMeierFitter
import plotly.io as pio
pio.renderers.default = "browser"

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"

def definitionTables_KM(data, filtrePLME=["PL", "ME"], plot=False):
    print("Execute tables_KM ---------------------")
    data = data[(data["PL_ME"].isin(filtrePLME)) | (data["PL_ME"] == "DD")]
    data.reset_index(drop=True, inplace=True)
    dataCER = data[data["type_ADH"] == "CER"]
    dataDP = data[data["type_ADH"] == "DP"]
    dataDPCER = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
    print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

    # Kaplan-Meier
    # Depuis l'état de cotisant :
    dataDP["age_sortie"] = np.array([np.select([dataDP["age_liq_rc"] > 0, dataDP["age_liq_rc"].isna()], [dataDP["age_liq_rc"], math.inf]), np.select([dataDP["age_rad"] > 0, dataDP["age_rad"].isna()], [dataDP["age_rad"], math.inf]), np.select([dataDP["age_dc"] > 0, dataDP["age_dc"].isna()], [dataDP["age_dc"], math.inf])]).min(axis=0)
    dataDP["age_derniereCot"] = np.array([dataDP["age_sortie"], dataDP["age"]]).min(axis=0)
    KM_duration_actif = pd.DataFrame(dataDP["age_derniereCot"] - dataDP["age_1ere_aff"])
    # actif to rad
    KM_observation_actiftoRad = pd.DataFrame(np.select([(dataDP["age_rad"] <= dataDP["age_liq_rc"]) | (dataDP["age_rad"] <= dataDP["age_dc"]) | ((dataDP["age_rad"] > 0) & (dataDP["age_liq_rc"].isna()) & (dataDP["age_dc"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_actif, event_observed=KM_observation_actiftoRad)
    if plot: kmf.survival_function_.plot(title="ACTIF to RAD")
    tableActiftoRad = 1 - kmf.survival_function_
    # actif to CER
    KM_observation_actiftoCER = pd.DataFrame(np.select([(dataDP["age_liq_rc"] <= dataDP["age_rad"]) | (dataDP["age_liq_rc"] <= dataDP["age_dc"]) | ((dataDP["age_liq_rc"] > 0) & (dataDP["age_rad"].isna()) & (dataDP["age_dc"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_actif, event_observed=KM_observation_actiftoCER)
    if plot: kmf.survival_function_.plot(title="ACTIF to CER")
    tableActiftoCER = 1 - kmf.survival_function_
    # actif to DC
    KM_observation_actiftoDC = pd.DataFrame(np.select([((dataDP["age_dc"] > 0) & (dataDP["age_rad"].isna()) & (dataDP["age_liq_rc"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_actif, event_observed=KM_observation_actiftoDC)
    if plot: kmf.survival_function_.plot(title="ACTIF to DC")
    tableActiftoDC = 1 - kmf.survival_function_
    # Depuis l'état de radié :
    dataDP_radie = dataDP[(dataDP["age_rad"] < dataDP["age_liq_rc"]) | ((dataDP["age_rad"] > 0) & (dataDP["age_liq_rc"].isna()))]
    dataDP_radie["age_sortie"] = np.array([np.select([dataDP_radie["age_liq_rc"] > 0, dataDP_radie["age_liq_rc"].isna()], [dataDP_radie["age_liq_rc"], math.inf]), np.select([dataDP_radie["age_dc"] > 0, dataDP_radie["age_dc"].isna()], [dataDP_radie["age_dc"], math.inf]), dataDP_radie["age"]]).min(axis=0)
    KM_duration_radie = pd.DataFrame(dataDP_radie["age_sortie"] - dataDP_radie["age_rad"])
    # rad to prest
    KM_observation_radtoPrest = pd.DataFrame(np.select([(dataDP_radie["age_liq_rc"] <= dataDP_radie["age_dc"]) | ((dataDP_radie["age_liq_rc"] > 0) & (dataDP_radie["age_dc"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_radie, event_observed=KM_observation_radtoPrest)
    if plot: kmf.survival_function_.plot(title="RAD to PREST")
    tableRadtoPrest = 1 - kmf.survival_function_
    # rad to DC
    KM_observation_radtoDC = pd.DataFrame(np.select([((dataDP_radie["age_dc"] > 0) & (dataDP_radie["age_liq_rc"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_radie, event_observed=KM_observation_radtoDC)
    if plot: kmf.survival_function_.plot(title="RAD to DC")
    tableRadtoDC = 1 - kmf.survival_function_
    # Depuis l'état de CER :
    dataDP_CER = dataDPCER[(dataDPCER["age_liq_rc"] < dataDPCER["age_rad"]) | ((dataDPCER["age_liq_rc"] > 0) & (dataDPCER["age_rad"].isna()))]
    dataDP_CER["age_sortie"] = np.array([np.select([dataDP_CER["age_rad"] > 0, dataDP_CER["age_rad"].isna()], [dataDP_CER["age_rad"], math.inf]), np.select([dataDP_CER["age_dc"] > 0, dataDP_CER["age_dc"].isna()], [dataDP_CER["age_dc"], math.inf]), dataDP_CER["age"]]).min(axis=0)
    KM_duration_CER = pd.DataFrame(dataDP_CER["age_sortie"] - dataDP_CER["age_liq_rc"])
    # CER to prest
    KM_observation_CERtoPrest = pd.DataFrame(np.select([(dataDP_CER["age_rad"] <= dataDP_CER["age_dc"]) | ((dataDP_CER["age_rad"] > 0) & (dataDP_CER["age_dc"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_CER, event_observed=KM_observation_CERtoPrest)
    if plot: kmf.survival_function_.plot(title="CER to PREST")
    tableCERtoPrest = 1 - kmf.survival_function_
    # CER to DC
    KM_observation_CERtoDC = pd.DataFrame(np.select([((dataDP_CER["age_dc"] > 0) & (dataDP_CER["age_rad"].isna()))], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_CER, event_observed=KM_observation_CERtoDC)
    if plot: kmf.survival_function_.plot(title="CER to DC")
    tableCERtoDC = 1 - kmf.survival_function_
    # Depuis l'état de PREST :
    dataDP_PREST = dataDPCER[(dataDPCER["age_liq_rc"] > 0) & (dataDPCER["age_rad"] > 0)]
    dataDP_PREST["age_sortie"] = np.array([np.select([dataDP_PREST["age_dc"] > 0, dataDP_PREST["age_dc"].isna()], [dataDP_PREST["age_dc"], math.inf]), dataDP_PREST["age"]]).min(axis=0)
    KM_duration_PREST = pd.DataFrame(dataDP_PREST["age_sortie"] - np.array([dataDP_PREST["age_liq_rc"], dataDP_PREST["age_rad"]]).max(axis=0))
    # PREST to DC
    KM_observation_PREST = pd.DataFrame(np.select([dataDP_PREST["age_dc"] > 0], [1]))
    kmf = KaplanMeierFitter()
    kmf.fit(KM_duration_PREST, event_observed=KM_observation_PREST)
    if plot: kmf.survival_function_.plot(title="PREST to DC")
    tablePRESTtoDC = 1 - kmf.survival_function_

    if plot:
        plt.figure(figsize=[16, 9])
        plt.title("Table de passage")
        plt.plot(tableActiftoRad, lebel="tableActiftoRad")
        plt.plot(tableActiftoCER, lebel="tableActiftoCER")
        plt.plot(tableActiftoDC, lebel="tableActiftoDC")
        plt.plot(tableRadtoPrest, lebel="tableRadtoPrest")
        plt.plot(tableRadtoDC, lebel="tableRadtoDC")
        plt.plot(tableCERtoPrest, lebel="tableCERtoPrest")
        plt.plot(tableCERtoDC, lebel="tableCERtoDC")
        plt.plot(tablePRESTtoDC, lebel="tablePRESTtoDC")
        plt.legend()

    print('--- FIN tables_KM --------------------------------------------------------------------------------------------------')
    return tableActiftoRad, tableActiftoCER, tableActiftoDC, tableRadtoPrest, tableRadtoDC, tableCERtoPrest, tableCERtoDC, tablePRESTtoDC






