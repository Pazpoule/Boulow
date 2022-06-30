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

def definitionTauxIntraAnnuel(data):
    dataDP = data[data["type_ADH"] == "DP"]
    tauxIntraAnnuel_RadtoPrest = len(dataDP[dataDP["age_rad"] == dataDP["age_liq_rc"]]) / len(dataDP[dataDP["age_rad"] > 0])
    tauxIntraAnnuel_RadtoDC = len(dataDP[dataDP["age_rad"] == dataDP["age_dc"]]) / len(dataDP[dataDP["age_rad"] > 0])
    dataCER = data[data["type_ADH"] == "CER"]
    tauxIntraAnnuel_CERtoPrest = len(dataCER[dataCER["age_rad"] == dataCER["age_liq_rc"]]) / len(dataCER[dataCER["age_liq_rc"] > 0])
    tauxIntraAnnuel_CERtoDC = len(dataCER[dataCER["age_dc"] == dataCER["age_liq_rc"]]) / len(dataCER[dataCER["age_liq_rc"] > 0])
    dataDPCER = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
    tauxIntraAnnuel_PresttoDC = len(dataDPCER[dataDPCER["age_dc"] == dataDPCER["age_liq_rc"]]) / len(dataDPCER[(dataDPCER["age_rad"] > 0) & (dataDPCER["age_liq_rc"] > 0)])
    return tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC, tauxIntraAnnuel_PresttoDC

def projectionAgesCOUNT(data, PL_ME, frequence, TGH05, TGF05, tauxIntraAnnuel_RadtoPrest, tauxIntraAnnuel_RadtoDC, tauxIntraAnnuel_PresttoDC, tauxIntraAnnuel_CERtoPrest, tauxIntraAnnuel_CERtoDC, difference_age_DD, tauxNuptialite, tauxReversion, ageMinReversion, lissageTaux = True, utiliserTableExp = False):
    COUNTActiftoRad, COUNTActiftoCER, COUNTRadtoPrest, COUNTCERtoPrest, COUNTDC, COUNTDC_H, COUNTDC_F, COUNTNul = definitionTables_COUNT(data, PL_ME, plot=True)
    dataDecede = data[data["age_dc"]>0]
    data = data[data["age_dc"].isna()]
    data["Type"] = data["Statut_ADH"]
    data = data.sample(frac=frequence) # échantilonne et mélange les individus
    data.reset_index(drop=True, inplace=True)
    tauxTable = "taux_lisse" if lissageTaux else "taux"

    baseNouveauDD = pd.DataFrame({})

    for adh in data.index:
        print(round(adh *100 /len(data), 2), ' %         projection ages COUNT')
        tableMortalite = TGF05 if data.loc[adh, 'Homme_Femme'] == 'F' else TGH05
        tableMortalite = pd.DataFrame({"taux": tableMortalite[data.loc[adh, 'an_nais']], "taux_lisse": tableMortalite[data.loc[adh, 'an_nais']]})
        tableDC = (COUNTDC_F if data.loc[adh, "Homme_Femme"] == "F" else COUNTDC_H) if utiliserTableExp else tableMortalite
        for age in range(int(data.loc[adh, 'age']), 120):
            if age != int(data.loc[adh, 'age']) or random.random() <= 0.5: # Suppose que la moitier des adh aujourd'hui ont déjà effectuer leur tirage pour leur age en cours
                tableRad = COUNTNul if (data.loc[adh, "Type"] in ["PRESTATAIRE", "RADIE", "DECEDE"]) else (COUNTCERtoPrest if data.loc[adh, "Type"] == "CER" else COUNTActiftoRad)
                tableLiq = COUNTNul if (data.loc[adh, "Type"] in ["CER", "PRESTATAIRE", "DECEDE"]) else (COUNTRadtoPrest if data.loc[adh, "Type"] == "RADIE" else COUNTActiftoCER)
                if data.loc[adh, "type_ADH"] == "DD":
                    tableRad = COUNTNul
                    tableLiq = COUNTNul
                passage = random.choices(["age_rad", "age_liq_rc", "age_dc", "age_encours"], [tableRad.loc[age, tauxTable], tableLiq.loc[age, tauxTable], tableDC.loc[age, tauxTable], 1- tableRad.loc[age, tauxTable] - tableLiq.loc[age, tauxTable] - tableDC.loc[age, tauxTable]])[0]
                data.loc[adh, passage] = age
                if passage == "age_rad" and data.loc[adh, "Type"] == "ACTIF":
                    data.loc[adh, "Type"] = "RADIE"
                    if random.random() <= tauxIntraAnnuel_RadtoPrest:
                        data.loc[adh, "age_liq_rc"] = age
                        data.loc[adh, "Type"] = "PRESTATAIRE"
                        data.loc[adh, "age_sortie"] = min((data.loc[adh, "age_liq_rc"] if data.loc[adh, "age_liq_rc"] > 0 else math.inf), (data.loc[adh, "age_rad"] if data.loc[adh, "age_rad"] > 0 else math.inf), (data.loc[adh, "age_dc"] if data.loc[adh, "age_dc"] > 0 else math.inf))
                        data.loc[adh, "PointsAccumule"] = data.loc[adh, "PointsCotiseParAn"] * (data.loc[adh, "age_sortie"] - data.loc[adh, "age_1ere_aff"])
                        if data.loc[adh, "PointsAccumule"] <= 180:
                            data.loc[adh, "BOOL_VERSEMENT_UNIQUE"] = 1
                    if random.random() <= tauxIntraAnnuel_RadtoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_rad" and data.loc[adh, "Type"] == "CER":
                    data.loc[adh, "Type"] = "PRESTATAIRE"
                    data.loc[adh, "age_sortie"] = min((data.loc[adh, "age_liq_rc"] if data.loc[adh, "age_liq_rc"] > 0 else math.inf), (data.loc[adh, "age_rad"] if data.loc[adh, "age_rad"] > 0 else math.inf), (data.loc[adh, "age_dc"] if data.loc[adh, "age_dc"] > 0 else math.inf))
                    data.loc[adh, "PointsAccumule"] = data.loc[adh, "PointsCotiseParAn"] * (data.loc[adh, "age_sortie"] - data.loc[adh, "age_1ere_aff"])
                    if data.loc[adh, "PointsAccumule"] <= 180:
                        data.loc[adh, "BOOL_VERSEMENT_UNIQUE"] = 1
                    if random.random() <= tauxIntraAnnuel_PresttoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_liq_rc" and data.loc[adh, "Type"] == "RADIE":
                    data.loc[adh, "Type"] = "PRESTATAIRE"
                    data.loc[adh, "age_sortie"] = min((data.loc[adh, "age_liq_rc"] if data.loc[adh, "age_liq_rc"] > 0 else math.inf), (data.loc[adh, "age_rad"] if data.loc[adh, "age_rad"] > 0 else math.inf), (data.loc[adh, "age_dc"] if data.loc[adh, "age_dc"] > 0 else math.inf))
                    data.loc[adh, "PointsAccumule"] = data.loc[adh, "PointsCotiseParAn"] * (data.loc[adh, "age_sortie"] - data.loc[adh, "age_1ere_aff"])
                    if data.loc[adh, "PointsAccumule"] <= 180:
                        data.loc[adh, "BOOL_VERSEMENT_UNIQUE"] = 1
                    if random.random() <= tauxIntraAnnuel_PresttoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_liq_rc" and data.loc[adh, "Type"] == "ACTIF":
                    data.loc[adh, "Type"] = "CER"
                    if random.random() <= tauxIntraAnnuel_CERtoPrest:
                        data.loc[adh, "age_rad"] = age
                        data.loc[adh, "Type"] = "PRESTATAIRE"
                        data.loc[adh, "age_sortie"] = min((data.loc[adh, "age_liq_rc"] if data.loc[adh, "age_liq_rc"] > 0 else math.inf), (data.loc[adh, "age_rad"] if data.loc[adh, "age_rad"] > 0 else math.inf), (data.loc[adh, "age_dc"] if data.loc[adh, "age_dc"] > 0 else math.inf))
                        data.loc[adh, "PointsAccumule"] = data.loc[adh, "PointsCotiseParAn"] * (data.loc[adh, "age_sortie"] - data.loc[adh, "age_1ere_aff"])
                        if data.loc[adh, "PointsAccumule"] <= 180:
                            data.loc[adh, "BOOL_VERSEMENT_UNIQUE"] = 1
                    if random.random() <= tauxIntraAnnuel_CERtoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_dc":
                    diff_age = random.choices(difference_age_DD["diff_age"], difference_age_DD["Nombre"])[0]
                    if data.loc[adh, "Type"] == "PRESTATAIRE" and data.loc[adh, "type_ADH"] != "DD" and age - diff_age >= ageMinReversion and random.random() <= tauxNuptialite: # Attention, minor le taux de nuptialité par la condition sur l'age
                        baseNouveauDD = baseNouveauDD.append({"type_ADH": "DD", "PL_ME": "DD", "ADH_NUM_ADHERENT": data.loc[adh, 'NUM_ADHERENT'], "Homme_Femme": "H" if (data.loc[adh, 'Homme_Femme'] == 'F') else "F", "an_nais": max(data.loc[adh, 'an_nais'] + diff_age, 1900), "age_liq_rc": age - diff_age, "age_dc": 0, "type_ADH": "DD", "NBR_POINT_REVERS_RC": data.loc[adh, 'PTS_RC_CARR']}, ignore_index=True)
                    data.loc[adh, "Type"] = "DECEDE"
                    break
    print('--- Projection des décès et liquidations: OK --------------------------------------------------------------------------------------------------')

    # Décès des nouveaux ayant-droit
    for adh in baseNouveauDD.index:
        print(round(adh*100/len(baseNouveauDD), 2), ' %         Boucle nouveaux réversataires (décès)')
        tableMortalite = TGF05 if baseNouveauDD.loc[adh, 'Homme_Femme'] == 'F' else TGH05
        tableMortalite = pd.DataFrame({"taux": tableMortalite[baseNouveauDD.loc[adh, 'an_nais']], "taux_lisse": tableMortalite[baseNouveauDD.loc[adh, 'an_nais']]})
        tableDC = (COUNTDC_F if baseNouveauDD.loc[adh, "Homme_Femme"] == "F" else COUNTDC_H) if utiliserTableExp else tableMortalite
        for age in range(int(baseNouveauDD.loc[adh, 'age_liq_rc']), 120):
            if random.random() <= tableDC.loc[age, tauxTable]:
                baseNouveauDD.loc[adh, "age_dc"] = age
                break
    data = data.append(baseNouveauDD)

    # On ajoute les adhérents déjà décédes
    data = data.append(dataDecede.sample(n=int(len(dataDecede)*frequence)))

    data.reset_index(drop=True, inplace=True)
    return data, baseNouveauDD









































