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
import Passif.Encode_input_data as inputData
import plotly.io as pio
pio.renderers.default = "browser"

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"


if __name__ == '__main__':

    PL_ME = ["PL"]
    debutProj = 2000
    finProj = 2070
    AnneeCalcul = 2021
    NombreNouveauxParAns = 6000
    VA = 46.96428571428571
    VS = 2.63
    Rendement = VS / VA
    tauxNuptialite = 0.4    # Nombre d'ayant droit (rattaché à un affilié) / Nombre d'affilié prestataire décédés
    tauxReversion = 0.54
    tauxRecouvrement = 0.95

    TGH05 = pd.read_excel(adresse + "Tables/TableMortaliteTGH05.xlsx", index_col=None, sheet_name='ProbaDC')
    TGF05 = pd.read_excel(adresse + "Tables/TableMortaliteTGF05.xlsx", index_col=None, sheet_name='ProbaDC')
    data = inputData.base
    difference_age_DD = pd.read_csv(adresse + 'Tables/difference_age_DD.csv', sep=';', encoding="ISO-8859-1")
    tauxPL = len(data[data["PL_ME"]=="PL"])/len(data)

    dataCER = data[data["type_ADH"] == "CER"]
    dataDP = data[data["type_ADH"] == "DP"]
    dataDPCER = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
    dataDD = data[data["type_ADH"] == "DD"]
    data = data[(data["PL_ME"].isin(PL_ME)) | (data["PL_ME"]=="DD")]
    data.reset_index(drop=True, inplace=True)
    print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

    tauxIntraAnnuel_RadtoPrest = len(dataDP[dataDP["age_rad"] == dataDP["age_liq_rc"]])/len(dataDP[dataDP["age_rad"]>0])
    tauxIntraAnnuel_RadtoDC = len(dataDP[dataDP["age_rad"] == dataDP["age_dc"]])/len(dataDP[dataDP["age_rad"]>0])
    tauxIntraAnnuel_CERtoPrest = len(dataCER[dataCER["age_rad"] == dataCER["age_liq_rc"]]) / len(dataCER[dataCER["age_liq_rc"] > 0])
    tauxIntraAnnuel_CERtoDC = len(dataCER[dataCER["age_dc"] == dataCER["age_liq_rc"]]) / len(dataCER[dataCER["age_liq_rc"] > 0])
    tauxIntraAnnuel_PresttoDC = len(dataDP[dataDPCER["age_dc"] == dataDPCER["age_liq_rc"]])/len(dataDPCER[(dataDPCER["age_rad"]>0) & (dataDPCER["age_liq_rc"]>0)])


    # Def des points
    data["age_sortie"] = np.array([np.select([data["age_liq_rc"] > 0, data["age_liq_rc"].isna()], [data["age_liq_rc"], math.inf]), np.select([data["age_rad"] > 0, data["age_rad"].isna()], [data["age_rad"], math.inf]), np.select([data["age_dc"] > 0, data["age_dc"].isna()], [data["age_dc"], math.inf])]).min(axis=0)
    data["age_derniereCot"] = np.array([data["age_sortie"], data["age"]]).min(axis=0)
    data["PointsCotiseParAn"] = data["PTS_RC_CARR"] / np.select([data["age_derniereCot"] - data["age_1ere_aff"]==0, data["age_derniereCot"] - data["age_1ere_aff"]!=0], [1, data["age_derniereCot"] - data["age_1ere_aff"]])
    pointsParAn = pd.DataFrame(data["PointsCotiseParAn"].value_counts())
    pointsParAn.sort_index(inplace=True)
    plt.figure(figsize=[16, 9])
    plt.title("Points")
    sns.distplot(pointsParAn, bins=1000, hist=True, kde=False, kde_kws={'shade': True, 'linewidth': 3}, label="Points")

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






    ageDC_H = pd.DataFrame(dataDPCER[dataDPCER["Homme_Femme"]=="H"]["age_dc"].value_counts())
    ageDC_H.sort_index(inplace=True)
    ageDC_H.columns = ["age_dc"]
    for indice in ageDC_H.index:
        ageDC_H.loc[indice, "assiette"] = len(dataDPCER[dataDPCER["Homme_Femme"]=="H"][(dataDPCER[dataDPCER["Homme_Femme"]=="H"]["age"] >= indice) & (dataDPCER[dataDPCER["Homme_Femme"]=="H"]["age_dc"] >= indice) & ((dataDPCER[dataDPCER["Homme_Femme"]=="H"]["age_1ere_aff"].isna()) | (dataDPCER[dataDPCER["Homme_Femme"]=="H"]["age_1ere_aff"] <= indice)) & (dataDPCER[dataDPCER["Homme_Femme"]=="H"]["age_dc"]>0)])
    ageDC_H["taux"] = ageDC_H["age_dc"] / ageDC_H["assiette"]
    ageDC_H.sort_index(inplace=True)
    ageDC_H.to_csv(adresse + f"Tables/Loi_dc{PL_ME}.csv")
    plt.plot(ageDC_H['taux'])
    
    ageDC_F = pd.DataFrame(dataDPCER[dataDPCER["Homme_Femme"]=="F"]["age_dc"].value_counts())
    ageDC_F.sort_index(inplace=True)
    ageDC_F.columns = ["age_dc"]
    for indice in ageDC_F.index:
        ageDC_F.loc[indice, "assiette"] = len(dataDPCER[dataDPCER["Homme_Femme"]=="F"][(dataDPCER[dataDPCER["Homme_Femme"]=="F"]["age"] >= indice) & (dataDPCER[dataDPCER["Homme_Femme"]=="F"]["age_dc"] >= indice) & ((dataDPCER[dataDPCER["Homme_Femme"]=="F"]["age_1ere_aff"].isna()) | (dataDPCER[dataDPCER["Homme_Femme"]=="F"]["age_1ere_aff"] <= indice)) & (dataDPCER[dataDPCER["Homme_Femme"]=="F"]["age_dc"]>0)])
    ageDC_F["taux"] = ageDC_F["age_dc"] / ageDC_F["assiette"]
    ageDC_F.sort_index(inplace=True)
    ageDC_F.to_csv(adresse + f"Tables/Loi_dc{PL_ME}.csv")
    plt.plot(ageDC_F['taux'])

    ageNul = pd.DataFrame({"taux" : [0 for i in range(0, 121)], "taux_lisse": [0 for i in range(0, 121)]})


    # -------------------------------------------------------------------------  Lissage et complétion à 0 des ages
    plt.figure(figsize=[16, 9])
    plt.title("Tables de passage lissé")
    def lissage(table, force=0.5):
        table["taux_lisse"] = table["taux"]
        for i, indice in enumerate(table.index[1:-1]):
            table.loc[indice, "taux_lisse"] = (1-force) * table.loc[indice, "taux"] + force * (table.loc[table.index[i], "taux"] + table.loc[table.index[i+2], "taux"])/2
        plt.plot(table["taux_lisse"])
        table = table.join(pd.DataFrame(index=[i for i in range(0, 121)]), how='outer').fillna(0)
        return table
    ageActiftoRad = lissage(ageActiftoRad)
    ageActiftoCER = lissage(ageActiftoCER)
    ageRadtoPrest = lissage(ageRadtoPrest)
    ageCERtoPrest = lissage(ageCERtoPrest)
    ageDC = lissage(ageDC)
    ageDC_H = lissage(ageDC_H)
    ageDC_F = lissage(ageDC_F)

    ageActiftoRad.to_csv(adresse + f"Tables/ageActiftoRad.csv")
    ageActiftoCER.to_csv(adresse + f"Tables/ageActiftoCER.csv")
    ageRadtoPrest.to_csv(adresse + f"Tables/ageRadtoPrest.csv")
    ageCERtoPrest.to_csv(adresse + f"Tables/ageCERtoPrest.csv")
    ageDC.to_csv(adresse + f"Tables/ageDC.csv")


    # On compare la table de mortalité nationnal pour l'année 1975 (génération moyenne)
    plt.figure(figsize=[16, 9])
    plt.title("Comparaison des tables de mortalité")
    data[data["age_dc"].isna()]["an_nais"].mean()
    tableMortalite = pd.DataFrame({"taux": TGH05[1975], "taux_lisse": TGH05[1975]})
    plt.plot(tableMortalite["taux_lisse"])
    plt.plot(ageDC_H["taux_lisse"])


    plt.figure(figsize=[16, 9])
    plt.title("Distribution des ages avant projection")
    distrib_aff = pd.DataFrame(data["age_1ere_aff"].value_counts())
    distrib_aff.sort_index(inplace=True)
    plt.plot(distrib_aff["age_1ere_aff"])
    distrib_liq = pd.DataFrame(data["age_liq_rc"].value_counts())
    distrib_liq.sort_index(inplace=True)
    plt.plot(distrib_liq["age_liq_rc"])
    distrib_dc = pd.DataFrame(data["age_dc"].value_counts())
    distrib_dc.sort_index(inplace=True)
    plt.plot(distrib_dc["age_dc"])

    dataDecede = data[data["age_dc"]>0]
    data = data[data["age_dc"].isna()]
    data["Type"] = data["Statut_ADH"]
    data.reset_index(drop=True, inplace=True)
    print('--- Pré-traitements des données: OK --------------------------------------------------------------------------------------------------')

    # On échantillonne pour aller plus vite
    frequence = 0.1
    lissageTaux = True
    utiliserTableExp = False
    tauxTable = "taux_lisse" if lissageTaux else "taux"
    dataComplet = data.copy()
    data = data.sample(frac = frequence)
    data.reset_index(drop=True, inplace=True)
    print('--- Echantillonnage: OK --------------------------------------------------------------------------------------------------')


    baseNouveauDD = pd.DataFrame({})

    for adh in range(len(data)):
        print(round(adh*100/len(data), 2), ' %         Boucle adhérants (décès et liq)')
        tableMortalite = TGF05 if data.loc[adh, 'Homme_Femme'] == 'F' else TGH05
        tableMortalite = pd.DataFrame({"taux": tableMortalite[data.loc[adh, 'an_nais']], "taux_lisse": tableMortalite[data.loc[adh, 'an_nais']]})
        tableDC = (ageDC_F if data.loc[adh, "Homme_Femme"] == "F" else ageDC_H) if utiliserTableExp else tableMortalite
        for age in range(int(data.loc[adh, 'age']), 120):
            if age != int(data.loc[adh, 'age']) or random.random() <= 0.5: # Suppose que la moitier des adh aujourd'hui ont déjà effectuer leur tirage pour leur age en cours
                tableRad = ageNul if (data.loc[adh, "Type"] in ["PRESTATAIRE", "RADIE", "DECEDE"]) else (ageCERtoPrest if data.loc[adh, "Type"] == "CER" else ageActiftoRad)
                tableLiq = ageNul if (data.loc[adh, "Type"] in ["CER", "PRESTATAIRE", "DECEDE"]) else (ageRadtoPrest if data.loc[adh, "Type"] == "RADIE" else ageActiftoCER)
                passage = random.choices(["age_rad", "age_liq_rc", "age_dc", "age_encours"], [tableRad.loc[age, tauxTable], tableLiq.loc[age, tauxTable], tableDC.loc[age, tauxTable], 1-tableRad.loc[age, tauxTable]-tableLiq.loc[age, tauxTable]-tableDC.loc[age, tauxTable]])[0]
                data.loc[adh, passage] = age
                if passage == "age_rad" and data.loc[adh, "Type"] == "ACTIF":
                    data.loc[adh, "Type"] = "RADIE"
                    if random.random() <= tauxIntraAnnuel_RadtoPrest:
                        data.loc[adh, "age_liq_rc"] = age
                        data.loc[adh, "Type"] = "PRESTATAIRE"
                        data.loc[adh, "age_sortie"] = min((data.loc[adh, "age_liq_rc"] if data.loc[adh, "age_liq_rc"]>0 else math.inf), (data.loc[adh, "age_rad"] if data.loc[adh, "age_rad"]>0 else math.inf), (data.loc[adh, "age_dc"] if data.loc[adh, "age_dc"]>0 else math.inf))
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
                        data.loc[adh, "age_sortie"] = min((data.loc[adh, "age_liq_rc"] if data.loc[adh, "age_liq_rc"]>0 else math.inf), (data.loc[adh, "age_rad"] if data.loc[adh, "age_rad"]>0 else math.inf), (data.loc[adh, "age_dc"] if data.loc[adh, "age_dc"]>0 else math.inf))
                        data.loc[adh, "PointsAccumule"] = data.loc[adh, "PointsCotiseParAn"] * (data.loc[adh, "age_sortie"] - data.loc[adh, "age_1ere_aff"])
                        if data.loc[adh, "PointsAccumule"] <= 180:
                            data.loc[adh, "BOOL_VERSEMENT_UNIQUE"] = 1
                    if random.random() <= tauxIntraAnnuel_CERtoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_dc":
                    if data.loc[adh, "Type"] == "PRESTATAIRE" and data.loc[adh, "type_ADH"] != "DD" and random.random() <= tauxNuptialite:
                        diff_age = random.choices(difference_age_DD["diff_age"], difference_age_DD["Nombre"])[0]
                        baseNouveauDD = baseNouveauDD.append({"Homme_Femme": "H" if (data.loc[adh, 'Homme_Femme'] == 'F') else "F", "an_nais": max(min(data.loc[adh, 'an_nais'] + diff_age, 1900), AnneeCalcul), "age_liq_rc": age + diff_age, "type_ADH" : "DD", "PTS_RC_CARR": tauxReversion * data.loc[adh, 'PTS_RC_CARR']}, ignore_index=True)
                    data.loc[adh, "Type"] = "DECEDE"
                    break
    print('--- Projection des décès et liquidations: OK --------------------------------------------------------------------------------------------------')

    # Décès des nouveaux ayant-droit
    for adh in range(len(baseNouveauDD)):
        print(round(adh*100/len(baseNouveauDD), 2), ' %         Boucle nouveaux réversataires (décès)')
        tableMortalite = TGF05 if baseNouveauDD.loc[adh, 'Homme_Femme'] == 'F' else TGH05
        tableMortalite = pd.DataFrame({"taux": tableMortalite[baseNouveauDD.loc[adh, 'an_nais']], "taux_lisse": tableMortalite[baseNouveauDD.loc[adh, 'an_nais']]})
        tableDC = (ageDC_F if baseNouveauDD.loc[adh, "Homme_Femme"] == "F" else ageDC_H) if utiliserTableExp else tableMortalite
        for age in range(int(baseNouveauDD.loc[adh, 'age_liq_rc']), 120):
            if random.random() <= tableDC.loc[age, tauxTable]:
                baseNouveauDD.loc[adh, "age_dc"] = age
                break
    data = data.append(baseNouveauDD)



    plt.figure(figsize=[16, 9])
    plt.title("Distribution des ages après projection")
    distrib_aff = pd.DataFrame(data["age_1ere_aff"].value_counts())
    distrib_aff.sort_index(inplace=True)
    plt.plot(distrib_aff["age_1ere_aff"])
    distrib_liq = pd.DataFrame(data["age_liq_rc"].value_counts())
    distrib_liq.sort_index(inplace=True)
    plt.plot(distrib_liq["age_liq_rc"])
    distrib_dc = pd.DataFrame(data["age_dc"].value_counts())
    distrib_dc.sort_index(inplace=True)
    plt.plot(distrib_dc["age_dc"])


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
    baseNewSample["an_1ere_aff"] = baseNewSample["an_nais"] + baseNewSample["age_1ere_aff"]
    print('--- Ajout New adh: OK ------------------------------------------------------------------------------------------------')



    plt.figure(figsize=[16, 9])
    plt.title("Distribution des ages après Nouveaux")
    distrib_aff = pd.DataFrame(data["age_1ere_aff"].value_counts())
    distrib_aff.sort_index(inplace=True)
    plt.plot(distrib_aff["age_1ere_aff"])
    distrib_liq = pd.DataFrame(data["age_liq_rc"].value_counts())
    distrib_liq.sort_index(inplace=True)
    plt.plot(distrib_liq["age_liq_rc"])
    distrib_dc = pd.DataFrame(data["age_dc"].value_counts())
    distrib_dc.sort_index(inplace=True)
    plt.plot(distrib_dc["age_dc"])

    # On ajoute les adhérents déjà décédes
    data = data.append(dataDecede) # DC non échantilloné
    data["age_sortie"] = np.array([np.select([data["age_liq_rc"] > 0, data["age_liq_rc"].isna()], [data["age_liq_rc"], math.inf]), np.select([data["age_rad"] > 0, data["age_rad"].isna()], [data["age_rad"], math.inf]), np.select([data["age_dc"] > 0, data["age_dc"].isna()], [data["age_dc"], math.inf])]).min(axis=0)
    data["age_derniereCot"] = np.array([data["age_sortie"], data["age"]]).min(axis=0)
    data["PointsAccumule"] = data["PointsCotiseParAn"] * (data["age_derniereCot"] - data["age_1ere_aff"])
    data.loc[data["PointsAccumule"] <= 180, "BOOL_VERSEMENT_UNIQUE"] = 1

    dataCER = data[data["type_ADH"]=="CER"]
    dataDP = data[data["type_ADH"]=="DP"]
    dataDD = data[data["type_ADH"]=="DD"]
    # On projete tout par année et on compte le nombre de cot et prest     (on suppose que l'on se place en fin d'année)
    baseProjection = pd.DataFrame({'Annee': [0 for annee in range(debutProj, finProj+1)], 'nbrCot':[0 for annee in range(debutProj, finProj+1)], 'nbrRadie':[0 for annee in range(debutProj, finProj+1)], 'nbrPrest':[0 for annee in range(debutProj, finProj+1)], 'nbrDC':[0 for annee in range(debutProj, finProj+1)], 'nbrDDnonLiq':[0 for annee in range(debutProj, finProj+1)], 'nbrDDPrest':[0 for annee in range(debutProj, finProj+1)]})
    for annee in range(debutProj, finProj + 1):
        print(round((annee-debutProj)*100/(finProj-debutProj), 2), ' %          Boucle projection des nombres adh par années')
        dataDP["age"] = annee - dataDP["an_nais"]
        dataCER["age"] = annee - dataCER["an_nais"]
        dataDD["age"] = annee - dataDD["an_nais"]
        baseProjection.loc[annee - debutProj, 'Annee'] = annee
        baseProjection.loc[annee - debutProj, 'nbrCot'] = len(dataDP[((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrRadie'] = len(dataDP[(dataDP['age_rad'] <= dataDP["age"]) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrest'] = len(dataDP[(dataDP["BOOL_VERSEMENT_UNIQUE"]==0) & ((dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrVFU'] = len(dataDP[(dataDP["BOOL_VERSEMENT_UNIQUE"]==1) & ((dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDC'] = len(dataDP[((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & (dataDP['age_dc'] <= dataDP["age"])])
        baseProjection.loc[annee - debutProj, 'nbrCotCER'] = len(dataCER[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrestCER'] = len(dataCER[(dataCER['age_rad'] <= dataCER["age"]) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDCCER'] = len(dataCER[dataCER['age_dc'] <= dataCER["age"]]) / frequence

    #  + dataDD[((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna()))]



        # todo integrer rid
        # todo inclure cer et dd dans proj fi
        # todo vfu 15x la pension (1/ancien rendement du point)
        baseProjection.loc[annee - debutProj, 'Cotisations'] = (tauxRecouvrement
                                                                * (sum(dataDP.loc[((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsCotiseParAn"])
                                                                + sum(dataCER.loc[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna())), "PointsCotiseParAn"]))
                                                                * VA) / frequence
        baseProjection.loc[annee - debutProj, 'Prestations'] = (tauxRecouvrement
                                                               * (sum(dataDP.loc[(dataDP["BOOL_VERSEMENT_UNIQUE"]==0) & ((dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))), "PointsAccumule"]) + sum(dataDD.loc[((dataDD['age_dc'] > dataDD["age"]) | (dataDD['age_dc'].isna())), "PointsAccumule"])
                                                               + sum(dataCER.loc[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna())), "PointsAccumule"])
                                                               + sum(dataCER.loc[(dataCER['age_rad'] <= dataCER["age"]) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna())), "PointsAccumule"]))
                                                               * VS
                                                               + sum(dataDP.loc[(dataDP["BOOL_VERSEMENT_UNIQUE"]==1) & ((dataDP['age_liq_rc'] == dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))), "PointsAccumule"])
                                                               * VA) / frequence

        baseProjection.loc[annee - debutProj, 'SoldeTechnique'] = baseProjection.loc[annee - debutProj, 'Cotisations'] - baseProjection.loc[annee - debutProj, 'Prestations']


    print('--- Projection par années: OK --------------------------------------------------------------------------------------------------')

    plt.figure(figsize=[16, 9])
    plt.title("Projection démographique")
    plt.plot(baseProjection["Annee"], baseProjection[['nbrCot', 'nbrRadie', 'nbrPrest', 'nbrDC']]) # Prest non CER
    plt.figure(figsize=[16, 9])
    plt.title("Projection CER")
    plt.plot(baseProjection["Annee"], baseProjection[['nbrCotCER', 'nbrPrestCER', 'nbrDCCER']])
    plt.figure(figsize=[16, 9])
    plt.title("Projection financière")
    plt.plot(baseProjection["Annee"], baseProjection[['Prestations', 'SoldeTechnique']])
    plt.plot(baseProjection["Annee"], baseProjection[['Cotisations']])


 # todo taux de recouvrement des cotisations



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

# baseProjection.to_csv(adresse + f"Output{PL_ME}.csv")






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






