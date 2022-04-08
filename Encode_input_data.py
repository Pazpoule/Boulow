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


def clean_data(base):
    base["age"] = AnneeCalcul - base["an_nais"]
    base["ageMinRtetraite"] = np.select([(base["an_nais"] < 1951), (base["an_nais"] == 1951), (base["an_nais"] == 1952), (base["an_nais"] == 1953), (base["an_nais"] == 1954), (base["an_nais"] >= 1955)], [60, 60.33, 60.75, 61.167, 61.584, 62])
    base["ageTauxPlein"] = np.select([(base["an_nais"] < 1951), (base["an_nais"] == 1951), (base["an_nais"] == 1952), (base["an_nais"] == 1953), (base["an_nais"] == 1954), (base["an_nais"] >= 1955)], [65, 65.33, 65.75, 66.167, 66.584, 67])
    base.loc[(base["Statut_ADH"]=="CER") & (base["an_liq_rc"].isna()), "Statut_ADH"] = "ACTIF"
    base.loc[base["an_dc"]>0, "Statut_ADH"] = "DECEDE"
    base.loc[(base["an_dc"].isna()) & (base["an_liq_rc"]>0) & (base["an_rad"]>0), "Statut_ADH"] = "PRESTATAIRE"
    base.loc[(base["an_dc"].isna()) & (base["an_liq_rc"].isna()) & (base["an_rad"]>0), "Statut_ADH"] = "RADIE"
    base.loc[(base["an_rad"].isna()) & (base["an_liq_rc"]>0) & (base["type_ADH"]!="DD"), "Statut_ADH"] = "CER"
    base.loc[base["Statut_ADH"]=="CER", "type_ADH"] = "CER"
    base.loc[(base["an_rad"].isna()) & (base["an_liq_rc"]>0) & (base["type_ADH"]!="DD"), "type_ADH"] = "CER"
    base.loc[base["an_liq_rc"] < base["an_rad"], "type_ADH"] = "CER"
    base["age_1ere_aff"] = base["an_1ere_aff"] - base["an_nais"]
    base["age_rad"] = base["an_rad"] - base["an_nais"]
    base["age_liq_rb"] = base["an_liq_rb"] - base["an_nais"]
    base["age_liq_rc"] = base["an_liq_rc"] - base["an_nais"]
    base["age_liq_id"] = base["an_liq_id"] - base["an_nais"]
    base["age_dc"] = base["an_dc"] - base["an_nais"]
    base_moins16 = base[(base["age"] < 16) & (base["type_ADH"] != "DD")]
    print(f"On compte {len(base_moins16)} affiliés de moins de 16 ans en droit propre, soit {round(100 * len(base_moins16) / len(base), 2)}%")
    base_plus120 = base[(base["age"] > 120) & (base["an_dc"].isna())]
    print(f"On compte {len(base_plus120)} affiliés de plus de 120 ans sans date de décès, soit {round(100 * len(base_plus120) / len(base), 2)}%")
    base_agesPlus120 = base[(base["age_1ere_aff"] > 120) | (base["age_rad"] > 120) | (base["age_liq_rc"] > 120) |(base["age_dc"] > 120)]
    print(f"On compte {len(base_agesPlus120)} affiliés dont les ages d'affiliation, radiation, liquidation, ou décès sont >120, soit {round(100 * len(base_agesPlus120) / len(base), 2)}%")
    base_agesMoins16 = base[((base["age_1ere_aff"] < 16) | (base["age_rad"] < 16) | (base["age_liq_rc"] < 16) | (base["age_dc"] < 16)) & (base["type_ADH"] != "DD")]
    print(f"On compte {len(base_agesMoins16)} affiliés en DP dont les ages d'affiliation, radiation, liquidation, ou décès sont <16, soit {round(100 * len(base_agesMoins16) / len(base), 2)}%")
    base_plus100nonLiq = base[(base["age"] > 100) & (base["an_dc"].isna()) & (base["an_liq_rc"].isna())]
    print(f"On compte {len(base_plus100nonLiq)} affiliés de plus de 100 ans sans date de décès ou date de liquidation, soit {round(100 * len(base_plus100nonLiq) / len(base), 2)}%")
    base_sansAgeDC = base[(base["an_dc"].isna()) & (base["DECEDE"] == 1)]
    print(f"On compte {len(base_sansAgeDC)} affiliés décèdé sans date de décès, soit {round(100 * len(base_sansAgeDC) / len(base), 2)}%")
    base_sansNais = base[base["an_nais"].isna()]
    print(f"On compte {len(base_sansNais)} affiliés sans date de Naissance, soit {round(100 * len(base_sansNais) / len(base), 2)}%")
    base_sansAff = base[(base["an_1ere_aff"].isna()) & (base["type_ADH"] != "DD")]
    print(f"On compte {len(base_sansAff)} affiliés DP sans date d'Affiliation, soit {round(100 * len(base_sansAff) / len(base), 2)}%")
    base_dcInfRad = base[base["an_dc"] < base["an_rad"]]
    print(f"On compte {len(base_dcInfRad)} affiliés décédés avant radiation, soit {round(100 * len(base_dcInfRad) / len(base), 2)}%")
    base_dcInfLiq = base[base["an_dc"] < base["an_liq_rc"]]
    print(f"On compte {len(base_dcInfLiq)} affiliés décédés avant liquidation, soit {round(100 * len(base_dcInfLiq) / len(base), 2)}%")
    base_dcInfAff = base[base["an_dc"] < base["an_1ere_aff"]]
    print(f"On compte {len(base_dcInfAff)} affiliés décédés avant affiliation, soit {round(100 * len(base_dcInfAff) / len(base), 2)}%")
    base_liqInfAff = base[base["an_liq_rc"] < base["an_1ere_aff"]]
    print(f"On compte {len(base_liqInfAff)} affiliés liquidés avant affiliation, soit {round(100 * len(base_liqInfAff) / len(base), 2)}%")
    base_radInfAff = base[base["an_rad"] < base["an_1ere_aff"]]
    print(f"On compte {len(base_radInfAff)} affiliés radiés avant affiliation, soit {round(100 * len(base_radInfAff) / len(base), 2)}%")
    base_AffInfNais = base[base["an_1ere_aff"] < base["an_nais"]]
    print(f"On compte {len(base_AffInfNais)} affiliés affiliés avant naissance, soit {round(100 * len(base_AffInfNais) / len(base), 2)}%")
    exclusion = pd.concat([base_moins16, base_plus120, base_agesPlus120, base_agesMoins16, base_plus100nonLiq, base_sansAgeDC, base_sansNais, base_sansAff, base_dcInfRad, base_dcInfLiq, base_dcInfAff, base_liqInfAff, base_radInfAff, base_AffInfNais])
    print(f"Au total, on exclu {len(exclusion)} affilié, soit {round(100 * len(exclusion) / len(base), 2)}%")
    base = pd.concat([base, exclusion]).drop_duplicates(keep=False)
    base.reset_index(drop=True, inplace=True)
    return base

def correct_data(base, dicoGroupement):
    print("On remplace les données manquantes par 0 là où ca a du sens")
    base.drop(["an_1ere_aff", "an_rad", "an_liq_rb", "an_liq_rc", "an_liq_id", "an_dc"], axis=1, inplace=True)
    base["BOOL_VERSEMENT_UNIQUE"] = base["BOOL_VERSEMENT_UNIQUE"].fillna(0)
    base["NBR_TRIM_CARR"] = base["NBR_TRIM_CARR"].fillna(0)
    base["PTS_RB_CARR"] = base["PTS_RB_CARR"].fillna(0)
    base["PTS_RC_CARR"] = base["PTS_RC_CARR"].fillna(0)
    for variable, seuil in dicoGroupement.items():
        print(f"On groupe les {seuil} {variable} les moins courentes dans une classe AUTRE")
        dataFrequence = pd.DataFrame(base[variable].value_counts())
        dataFrequence.reset_index(inplace=True)
        aGrouper = dataFrequence[dataFrequence[variable] < seuil]
        dataAGrouper = base[base[variable].isin(aGrouper["index"])]
        dataAGrouper[variable] = "AUTRE"
        base = base[~base[variable].isin(aGrouper["index"])].append(dataAGrouper)
        base[variable] = base[variable].fillna("AUTRE")
    base.reset_index(drop=True, inplace=True)
    return base


print("Execute encode_input_data ---------------------")

outputStat = False
AnneeCalcul = 2021
seuilLourdLeger = 9
dicoGroupement = {"profession": 2000, "cp": 300}

TGH05 = pd.read_excel(adresse + "Tables/TableMortaliteTGH05.xlsx", index_col=None, sheet_name='ProbaDC')
TGF05 = pd.read_excel(adresse + "Tables/TableMortaliteTGF05.xlsx", index_col=None, sheet_name='ProbaDC')
data = pd.read_csv(adresse + 'Tables/DATA.csv', sep=';', encoding="ISO-8859-1")
print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

print("Traitement donnée complète -------------------------------------------------------------------")
base = clean_data(data)
base = correct_data(base, dicoGroupement)

# Import la base prest pour récupérer les VFU
dataPrest = pd.read_csv(adresse + 'Tables/basePrest.csv', sep=';', encoding="ISO-8859-1")
dataPrest["type_ADH"] = "DP"
dataPrest.loc[dataPrest["COTISANT"]==1, "type_ADH"] = "CER"
dataPrest = clean_data(dataPrest)

base = base.merge(dataPrest[["NUM_ADHERENT_FONCT", "BLOCAGE", "MOTIF_BLOCAGE", "POINT_PAYE_RC", "POINT_ACQUI_RC", "POINT_ACQUI_ID", "POINT_PAYE_ID"]], on="NUM_ADHERENT_FONCT", how="left").drop_duplicates()
base.loc[base["BLOCAGE"] == 1, "BOOL_VERSEMENT_UNIQUE"] = 1
base.loc[base["type_ADH"] == "DD", "PTS_RC_CARR"] = base.loc[base["type_ADH"] == "DD", "NBR_POINT_REVERS_RC"]
base.loc[base["type_ADH"] == "DD", "PL_ME"] = "DD"

# On récupère les points ID par classe
basePrestID_A = base[(base["age_liq_id"]>0) & ((base["POINT_ACQUI_ID"]<=200) | (base["CLASSE_CHOISIE_ID"]=="A"))]
nbrPointPayeID_A = basePrestID_A["POINT_PAYE_ID"].mean()
basePrestID_B = base[(base["age_liq_id"]>0) & (((base["POINT_ACQUI_ID"]>200) & (base["POINT_ACQUI_ID"]<=600)) | (base["CLASSE_CHOISIE_ID"]=="B"))]
nbrPointPayeID_B = basePrestID_B["POINT_PAYE_ID"].mean()
basePrestID_C = base[(base["age_liq_id"]>0) & (((base["POINT_ACQUI_ID"]>600) & (base["POINT_ACQUI_ID"]<=1000)) | (base["CLASSE_CHOISIE_ID"]=="C"))]
nbrPointPayeID_C = basePrestID_C["POINT_PAYE_ID"].mean()
proportionClassesID = [len(basePrestID_A), len(basePrestID_B), len(basePrestID_C)]





















# baseDD = base[(base["ADH_NUM_ADHERENT"].isin(base["NUM_ADHERENT"])) & (base["ADH_NUM_ADHERENT"]>0)][["ADH_NUM_ADHERENT", "age"]].drop_duplicates()
# baseDD.drop(["BOOL_VERSEMENT_UNIQUE"], axis=1, inplace=True)
# baseDD.drop_duplicates(inplace=True)
# baseDP = base[(base["NUM_ADHERENT"].isin(base["ADH_NUM_ADHERENT"])) & (base["NUM_ADHERENT"]>0)].drop_duplicates()
# baseDD.sort_values(by="ADH_NUM_ADHERENT", inplace=True)
# test = baseDD[baseDD["ADH_NUM_ADHERENT"]==295273]
# base["age_dd"] = np.where((base["NUM_ADHERENT"].isin(base["ADH_NUM_ADHERENT"])) & (base["NUM_ADHERENT"]>0), )

# # On test les intersection entre la base COT et la base PREST
# Prest1 = base[((base["Statut_ADH"]=="PRESTATAIRE") | (base["Statut_ADH"]=="CER")) & (base["type_ADH"]!="DD") & (base["BOOL_VERSEMENT_UNIQUE"]!=1) & (base["BLOCAGE"]==0)].drop_duplicates()
# Prest2 = dataPrest[(dataPrest["an_dc"].isna()) & (dataPrest["DECEDE"]==0) & (dataPrest["BLOCAGE"]==0)].drop_duplicates()
# _1ET2 = pd.merge(Prest1, Prest2, on="NUM_ADHERENT_FONCT", how='inner').drop_duplicates()
# _1OU2 = pd.merge(Prest1, Prest2, on="NUM_ADHERENT_FONCT", how="outer").drop_duplicates()
# _1non2 = Prest1[~Prest1["NUM_ADHERENT_FONCT"].isin(list(Prest2["NUM_ADHERENT_FONCT"]))].drop_duplicates()
# _2non1 = Prest2[~Prest2["NUM_ADHERENT_FONCT"].isin(list(Prest1["NUM_ADHERENT_FONCT"]))].drop_duplicates()
# Prest1["POINT_PAYE_RC"].sum()
# Prest1["PTS_RC_CARR"].sum()

if outputStat:

    # On def la baseStat que l'on va utiliser (pour ne pas modifier la base d'export)
    baseStat = base.copy()
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






    #
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












def encode_data_NN(base):
    # oneHot_sexe = pd.get_dummies(base.Homme_Femme, prefix='sexe')
    oneHot_profession = pd.get_dummies(base.profession, prefix='profession')
    oneHot_cp = pd.get_dummies(base.cp, prefix='lieu')
    # oneHot_PLME = pd.get_dummies(base.PL_ME, prefix='PLME')
    oneHot_statut = pd.get_dummies(base.Statut_ADH, prefix='statut')
    listMoyenne = []
    listSD = []
    for colonne in ["an_nais", "an_1ere_aff", "an_liq_rc", "an_rad", "an_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        listMoyenne.append(base[colonne].mean())
        listSD.append(base[colonne].std())
    base_NN = base[["an_nais", "an_1ere_aff", "an_liq_rc", "an_rad", "an_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]]
    nbrVarNumeriques = len(base_NN.columns)
    base_NN["Sexe"] = 0
    base_NN.loc[base["Homme_Femme"] == "F", "Sexe"] = 1
    base_NN["PL"] = 0
    base_NN.loc[base["PL_ME"] == "PL", "PL"] = 1
    base_NN = base_NN.join(oneHot_profession.join(oneHot_cp.join(oneHot_statut)))
    base_NN["BOOL_VERSEMENT_UNIQUE"] = base["BOOL_VERSEMENT_UNIQUE"].astype('int64')
    nbrVarCategorielles = len(base_NN.columns) - nbrVarNumeriques
    for colonne in range(nbrVarCategorielles):
        listMoyenne.append(0)
        listSD.append(1)
    base_NN = (base_NN - listMoyenne) / listSD

    # Pour DC
    NN_dc = base_NN
    listMoyenne_dc = listMoyenne
    listSD_dc = listSD
    NN_dc["bool_radie"] = np.where(NN_dc['an_rad'] > 0, 1, 0)
    listMoyenne_dc.append(0)
    listSD_dc.append(1)
    NN_dc["bool_liq"] = np.where(NN_dc['an_liq_rc'] > 0, 1, 0)
    listMoyenne_dc.append(0)
    listSD_dc.append(1)
    NN_dc["an_rad"] = NN_dc["an_rad"].fillna(NN_dc["an_rad"].mean())
    NN_dc["an_liq_rc"] = NN_dc["an_liq_rc"].fillna(NN_dc["an_liq_rc"].mean())
    NN_dc = NN_dc[NN_dc["an_dc"] > 0]
    X = list(NN_dc.columns)
    X.pop(4)
    listMoyenne_dc.pop(4)
    listSD_dc.pop(4)
    NN_dc = NN_dc[X + ["an_dc"]]
    listMoyenne_dc.append(listMoyenne[4])
    listSD_dc.append(listSD[4])

    # Pour Liq
    NN_liq = base_NN
    listMoyenne_liq = listMoyenne
    listSD_liq = listSD
    NN_liq["bool_radie"] = np.where(NN_liq['an_rad'] > 0, 1, 0)
    listMoyenne_liq.append(0)
    listSD_liq.append(1)
    NN_liq["an_rad"] = NN_liq["an_rad"].fillna(NN_liq["an_rad"].mean())
    NN_liq = NN_liq[NN_liq["an_liq"] > 0]
    X = list(NN_liq.columns)
    # exclure le statut
    X.pop(2)
    X.pop(4)

    NN_liq = NN_liq[X + ["an_liq"]]

























































