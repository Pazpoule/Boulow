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
# import plotly.io as pio
# pio.renderers.default = "browser"

def lissage(table, force=0.5):
    for colonne in table.columns:
        table[colonne+"_lisse"] = table[colonne]
        for i, indice in enumerate(table.index[1:-1]):
            table.loc[indice, colonne+"_lisse"] = (1 - force) * table.loc[indice, colonne] + force * (table.loc[table.index[i], colonne] + table.loc[table.index[i + 2], colonne]) / 2
        table[colonne] = table[colonne+"_lisse"]
        table.drop([colonne+"_lisse"], axis=1)
    return table

def volumeAnnuel(data, frequence, debutProj, finProj, tauxRecouvrement, VA, VS, dataEvolutionDuPoint, coeffVFU, tauxReversion, dataCotisation, plot=True, lisse=False):
    for colonne in ["age_liq_rb", "age_liq_id", "NBR_POINT_REVERS_RC", "bool_liq_rc", "bool_rad", "bool_dc"]:
        if colonne not in data.columns:
            data[colonne] = math.inf
    print("Execute VolumeAnnuel ---------------------")
    # On découpe la donnée en trois tables : les DD, les DP et les CER (qui sont aussi des DP mais que l'on traite séparément)
    dataCER = data[data["type_ADH"] == "CER"]
    dataDP = data[data["type_ADH"] == "DP"]
    dataDD = data[data["type_ADH"] == "DD"]
    # On projete tout par année et on compte le nombre de cot et prest     (on suppose que l'on se place en fin d'année)
    baseProjection = pd.DataFrame({'Annee': [0 for annee in range(debutProj, finProj + 1)], 'nbrCot': [0 for annee in range(debutProj, finProj + 1)], 'nbrRadie': [0 for annee in range(debutProj, finProj + 1)], 'nbrPrest': [0 for annee in range(debutProj, finProj + 1)], 'nbrDC': [0 for annee in range(debutProj, finProj + 1)], 'nbrDD': [0 for annee in range(debutProj, finProj + 1)], 'nbrDDDC': [0 for annee in range(debutProj, finProj + 1)]})
    for annee in range(debutProj, finProj + 1):
        print(round((annee - debutProj) * 100 / (finProj - debutProj), 2), ' %          Boucle projection des volumes annuels')
        dataDP["age"] = annee - dataDP["an_nais"]
        dataCER["age"] = annee - dataCER["an_nais"]
        dataDD["age"] = annee - dataDD["an_nais"]
        baseProjection.loc[annee - debutProj, 'Annee'] = annee
        # Projections démographiques générales
        baseProjection.loc[annee - debutProj, 'nbrCot'] = len(dataDP[((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrRadie'] = len(dataDP[(dataDP['age_rad'] <= dataDP["age"]) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrest'] = len(dataDP[(dataDP["BOOL_VERSEMENT_UNIQUE"] == 0) & (dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        # il arrive que certains VFU n'ai pas de date de liq. il ne devraient pas ^tre considéré comme prest.
        baseProjection.loc[annee - debutProj, 'nbrVFU'] = len(dataDP[(dataDP["BOOL_VERSEMENT_UNIQUE"] == 1) & (dataDP["age_liq_rc"] <= dataDP["age"]) & (dataDP["bool_liq_rc"]!=0)]) / frequence # cumulatif
        baseProjection.loc[annee - debutProj, 'nbrDC'] = len(dataDP[dataDP['age_dc'] <= dataDP["age"]]) / frequence # cumulatif
        baseProjection.loc[annee - debutProj, 'nbrCotCER'] = len(dataCER[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrestCER'] = len(dataCER[(dataCER['age_rad'] <= dataCER["age"]) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDCCER'] = len(dataCER[dataCER['age_dc'] <= dataCER["age"]]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDD'] = len(dataDD[((dataDD["age_liq_rb"] <= dataDD["age"]) | (dataDD["age_liq_rc"] <= dataDD["age"]) | (dataDD["age_liq_id"] <= dataDD["age"])) & ((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDDDC'] = len(dataDD[(dataDD['age_dc'] <= dataDD["age"]) | (dataDD['Statut_ADH'] == "DECEDE")]) / frequence

        # # todo integrer rid
        # Projections financières générales
        baseProjection.loc[annee - debutProj, 'Cotisations'] = (tauxRecouvrement
                                                                * (sum(dataDP.loc[((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & (dataDP['age_1ere_aff'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsCotiseParAn"])
                                                                   + sum(dataCER.loc[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & (dataCER['age_1ere_aff'] <= dataCER["age"]) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna())), "PointsCotiseParAn"]))
                                                                * (dataEvolutionDuPoint.loc[annee, "VA_modif"] if annee in dataEvolutionDuPoint.index else VA)) / frequence
        baseProjection.loc[annee - debutProj, 'Prestations'] = (tauxRecouvrement
                                                                * (sum(dataDP.loc[(dataDP["BOOL_VERSEMENT_UNIQUE"] == 0) & (dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsAccumule"])
                                                                   + sum(dataDD.loc[((dataDD["age_liq_rb"] <= dataDD["age"]) | (dataDD["age_liq_rc"] <= dataDD["age"]) | (dataDD["age_liq_id"] <= dataDD["age"])) & ((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna())), "NBR_POINT_REVERS_RC"]) * tauxReversion # On suppose que les points reversé en base sont avant taux de réversion. A verifier
                                                                   + sum(dataCER.loc[(dataCER['age_liq_rc'] <= dataCER["age"]) & (dataCER['age_1ere_aff'] <= dataCER["age"]) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna())), "PointsAccumule"]))
                                                                * (dataEvolutionDuPoint.loc[annee, "VS_modif"] if annee in dataEvolutionDuPoint.index else VS)
                                                                + sum(dataDP.loc[(dataDP["BOOL_VERSEMENT_UNIQUE"] == 1) & (dataDP['age_liq_rc'] == dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsAccumule"])
                                                                * (dataEvolutionDuPoint.loc[annee, "VS_modif"] if annee in dataEvolutionDuPoint.index else VS) * coeffVFU
                                                                ) / frequence
        baseProjection.loc[annee - debutProj, 'SoldeTechnique'] = baseProjection.loc[annee - debutProj, 'Cotisations'] - baseProjection.loc[annee - debutProj, 'Prestations']

        # Projections démographiques DSS
        baseProjection.loc[annee - debutProj, 'nbrCot_H'] = len(dataDP[(dataDP['Homme_Femme'] == "H") & ((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrCot_F'] = len(dataDP[(dataDP['Homme_Femme'] == "F") & ((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrRadie_H'] = len(dataDP[(dataDP['Homme_Femme'] == "H") & (dataDP['age_rad'] <= dataDP["age"]) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrRadie_F'] = len(dataDP[(dataDP['Homme_Femme'] == "F") & (dataDP['age_rad'] <= dataDP["age"]) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrest_H'] = len(dataDP[(dataDP['Homme_Femme'] == "H") & (dataDP["BOOL_VERSEMENT_UNIQUE"] == 0) & (dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrest_F'] = len(dataDP[(dataDP['Homme_Femme'] == "F") & (dataDP["BOOL_VERSEMENT_UNIQUE"] == 0) & (dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrVFU_H'] = len(dataDP[(dataDP['Homme_Femme'] == "H") & (dataDP["BOOL_VERSEMENT_UNIQUE"] == 1) & (dataDP['age_liq_rc'] <= dataDP["age"])]) / frequence # cumulatif
        baseProjection.loc[annee - debutProj, 'nbrVFU_F'] = len(dataDP[(dataDP['Homme_Femme'] == "F") & (dataDP["BOOL_VERSEMENT_UNIQUE"] == 1) & (dataDP['age_liq_rc'] <= dataDP["age"])]) / frequence # cumulatif
        baseProjection.loc[annee - debutProj, 'nbrCotCER'] = len(dataCER[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrestCER'] = len(dataCER[(dataCER['age_rad'] <= dataCER["age"]) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDCCER'] = len(dataCER[dataCER['age_dc'] <= dataCER["age"]]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDD_H'] = len(dataDD[(dataDD['Homme_Femme'] == "H") & ((dataDD["age_liq_rb"] <= dataDD["age"]) | (dataDD["age_liq_rc"] <= dataDD["age"]) | (dataDD["age_liq_id"] <= dataDD["age"])) & ((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDD_F'] = len(dataDD[(dataDD['Homme_Femme'] == "F") & ((dataDD["age_liq_rb"] <= dataDD["age"]) | (dataDD["age_liq_rc"] <= dataDD["age"]) | (dataDD["age_liq_id"] <= dataDD["age"])) & ((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'Prestations_DP'] = (tauxRecouvrement
                                                                * (sum(dataDP.loc[(dataDP["BOOL_VERSEMENT_UNIQUE"] == 0) & (dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsAccumule"])
                                                                   + sum(dataCER.loc[(dataCER['age_liq_rc'] <= dataCER["age"]) & (dataCER['age_1ere_aff'] <= dataCER["age"]) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna())), "PointsAccumule"]))
                                                                * (dataEvolutionDuPoint.loc[annee, "VS_modif"] if annee in dataEvolutionDuPoint.index else VS)
                                                                ) / frequence
        baseProjection.loc[annee - debutProj, 'Prestations_DD'] = (tauxRecouvrement
                                                                * sum(dataDD.loc[((dataDD["age_liq_rb"] <= dataDD["age"]) | (dataDD["age_liq_rc"] <= dataDD["age"]) | (dataDD["age_liq_id"] <= dataDD["age"])) & ((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna())), "NBR_POINT_REVERS_RC"]) * tauxReversion
                                                                * (dataEvolutionDuPoint.loc[annee, "VS_modif"] if annee in dataEvolutionDuPoint.index else VS)
                                                                ) / frequence
        baseProjection.loc[annee - debutProj, 'Prestations_VFU'] = (tauxRecouvrement
                                                                * sum(dataDP.loc[(dataDP["BOOL_VERSEMENT_UNIQUE"] == 1) & (dataDP['age_liq_rc'] == dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsAccumule"])
                                                                * (dataEvolutionDuPoint.loc[annee, "VS_modif"] if annee in dataEvolutionDuPoint.index else VS) * coeffVFU
                                                                ) / frequence
    if lisse:
        baseProjection = lissage(baseProjection)
    if plot:
        plt.figure(figsize=[16, 9])
        plt.title("Projection démographique")
        plt.plot(baseProjection["Annee"], baseProjection[['nbrCot', 'nbrRadie', 'nbrPrest', 'nbrDC', 'nbrVFU']], label = ['Cotisants', 'Radiés', 'Prestataires', 'Décédés', "VFU"])  # Prest non CER
        plt.legend()
        plt.figure(figsize=[16, 9])
        plt.title("Projection CER")
        plt.plot(baseProjection["Annee"], baseProjection[['nbrCotCER', 'nbrPrestCER', 'nbrDCCER']], label = ["Cotisants CER", "Prestataires CER", "Décédés CER"])
        plt.legend()
        plt.figure(figsize=[16, 9])
        plt.title("Projection financière")
        plt.plot(baseProjection["Annee"], baseProjection[['Cotisations', 'Prestations_DP', 'Prestations_DD', 'Prestations_VFU', 'SoldeTechnique']], label = ['Cotisations', 'Prestations_DP', 'Prestations_DD', 'Prestations_VFU', 'SoldeTechnique'])
        plt.legend()
    print('--- FIN VolumeAnnuel --------------------------------------------------------------------------------------------------')
    return baseProjection






    # TODO affiner la mortalité


    # TODo régime ID (et gérer les DD id)
    # TODo gérer entré DD - Les adh dans la baseDD sont tous liquidés, soit rb, rc ou id
    # TODo DC CER sous-estimé
    # TODo def loi sur pl puis me
    # TODo vfu des DD (cf doc des reversation sur tempo)


    # TODO hyp de rendement fi
    # TODO integrer la mise en asif de l'age min historique -> nouvelle hyp de passage
    # TODO découpage plfss
    # TODO

    # TODo projeter le point
    # TODO décalage de l'age de retraite
    # TODO hyp sur le revenu
    # TODO hyp sur l'inflation
    # TODO choc d'hypothèse de passage
    # TODO hyp de réduction des ne-liq_jms


















































































