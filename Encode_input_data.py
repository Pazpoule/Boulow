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

def clean_data(base, anneeCalcul):
    base["age"] = AnneeCalcul - base["an_nais"]
    for annee in ["an_liq_rb", "an_liq_rc", "an_liq_id", "an_rad", "an_dc"]: # on vire toutes les données trop récente
        base.loc[base[annee]>anneeCalcul, annee] = np.nan
    base["ageMinRtetraite"] = np.select([(base["an_nais"] < 1951), (base["an_nais"] == 1951), (base["an_nais"] == 1952), (base["an_nais"] == 1953), (base["an_nais"] == 1954), (base["an_nais"] >= 1955)], [60, 60.33, 60.75, 61.167, 61.584, 62])
    base["ageTauxPlein"] = np.select([(base["an_nais"] < 1951), (base["an_nais"] == 1951), (base["an_nais"] == 1952), (base["an_nais"] == 1953), (base["an_nais"] == 1954), (base["an_nais"] >= 1955)], [65, 65.33, 65.75, 66.167, 66.584, 67])
    base.loc[(base["Statut_ADH"] == "CER") & (base["an_liq_rc"].isna()), "Statut_ADH"] = "ACTIF"
    base.loc[base["an_dc"]>0, "Statut_ADH"] = "DECEDE"
    base.loc[(base["an_dc"].isna()) & (base["an_liq_rc"]>0) & (base["an_rad"]>0), "Statut_ADH"] = "PRESTATAIRE"
    base.loc[(base["an_dc"].isna()) & (base["an_liq_rc"].isna()) & (base["an_rad"]>0), "Statut_ADH"] = "RADIE"
    base.loc[(base["an_rad"].isna()) & (base["an_liq_rc"]>0) & (base["type_ADH"]!="DD"), "Statut_ADH"] = "CER"
    base.loc[base["Statut_ADH"] == "CER", "type_ADH"] = "CER"
    base.loc[(base["an_rad"].isna()) & (base["an_liq_rc"]>0) & (base["type_ADH"]!="DD"), "type_ADH"] = "CER"
    base.loc[base["an_liq_rc"] < base["an_rad"], "type_ADH"] = "CER"
    base["age_1ere_aff"] = base["an_1ere_aff"] - base["an_nais"]
    base["age_rad"] = base["an_rad"] - base["an_nais"]
    base["age_liq_rb"] = base["an_liq_rb"] - base["an_nais"]
    base["age_liq_rc"] = base["an_liq_rc"] - base["an_nais"]
    base["age_liq_id"] = base["an_liq_id"] - base["an_nais"]
    base["age_dc"] = base["an_dc"] - base["an_nais"]
    base.loc[base["Homme_Femme"] == "M", "Homme_Femme"] = "H"
    base_tropRecent = base[base["an_1ere_aff"] > anneeCalcul]
    print(f"On compte {len(base_tropRecent)} affiliés apres l'année de calcul, soit {round(100 * len(base_tropRecent) / len(base), 2)}%")
    base_sexeNA = base[base["Homme_Femme"].isna()]
    print(f"On compte {len(base_sexeNA)} affiliés dont le sexe est manquant, soit {round(100 * len(base_sexeNA) / len(base), 2)}%")
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
    exclusion = pd.concat([base_tropRecent, base_sexeNA, base_moins16, base_plus120, base_agesPlus120, base_agesMoins16, base_plus100nonLiq, base_sansAgeDC, base_sansNais, base_sansAff, base_dcInfRad, base_dcInfLiq, base_dcInfAff, base_liqInfAff, base_radInfAff, base_AffInfNais])
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
    base["NBR_POINT_REVERS_RC"] = base["NBR_POINT_REVERS_RC"].fillna(0)
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

AnneeCalcul = 2021
dicoGroupement = {"profession": 2000, "cp": 300}
data = pd.read_csv(adresse + 'Tables/DATA.csv', sep=';', encoding="ISO-8859-1")
print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

print("Traitement donnée complète -------------------------------------------------------------------")
base = clean_data(data, AnneeCalcul)
base = correct_data(base, dicoGroupement)

# Import la base prest pour récupérer les VFU
dataPrest = pd.read_csv(adresse + 'Tables/basePrest.csv', sep=';', encoding="ISO-8859-1")
dataPrest["type_ADH"] = "DP"
dataPrest.loc[dataPrest["COTISANT"] == 1, "type_ADH"] = "CER"
dataPrest = clean_data(dataPrest, AnneeCalcul)

base = base.merge(dataPrest[["NUM_ADHERENT_FONCT", "BLOCAGE", "MOTIF_BLOCAGE", "POINT_PAYE_RC", "POINT_ACQUI_RC", "POINT_ACQUI_ID", "POINT_PAYE_ID"]], on="NUM_ADHERENT_FONCT", how="left").drop_duplicates()
base.loc[base["BLOCAGE"] == 1, "BOOL_VERSEMENT_UNIQUE"] = 1
base.loc[base["type_ADH"] == "DD", "PTS_RC_CARR"] = base.loc[base["type_ADH"] == "DD", "NBR_POINT_REVERS_RC"]
base.loc[base["type_ADH"] == "DD", "PL_ME"] = "DD"

# Def des points
base["age_sortie"] = np.array([np.select([base["age_liq_rc"] > 0, base["age_liq_rc"].isna()], [base["age_liq_rc"], math.inf]), np.select([base["age_rad"] > 0, base["age_rad"].isna()], [base["age_rad"], math.inf]), np.select([base["age_dc"] > 0, base["age_dc"].isna()], [base["age_dc"], math.inf])]).min(axis=0)
base["age_derniereCot"] = np.array([base["age_sortie"], base["age"]]).min(axis=0)
base["PointsCotiseParAn"] = base["PTS_RC_CARR"] / np.select([base["age_derniereCot"] - base["age_1ere_aff"] == 0, base["age_derniereCot"] - base["age_1ere_aff"] != 0], [1, base["age_derniereCot"] - base["age_1ere_aff"]])
base["PointsCotiseParAn"] = base["PointsCotiseParAn"].fillna(0)
base["PointsAccumule"] = base["PTS_RC_CARR"].fillna(0)
# pointsParAn = pd.DataFrame(base["PointsCotiseParAn"].value_counts())
# pointsParAn.sort_index(inplace=True)
# plt.figure(figsize=[16, 9])
# plt.title("Points")
# sns.distplot(pointsParAn, bins=1000, hist=True, kde=False, kde_kws={'shade': True, 'linewidth': 3}, label="Points")

base.to_csv(adresse + f"Tables/DATA_nettoye.csv", index=False)
print("FIN encode_input_data ---------------------")

# # On récupère les points ID par classe
# basePrestID_A = base[(base["age_liq_id"]>0) & ((base["POINT_ACQUI_ID"]<=200) | (base["CLASSE_CHOISIE_ID"]=="A"))]
# nbrPointPayeID_A = basePrestID_A["POINT_PAYE_ID"].mean()
# basePrestID_B = base[(base["age_liq_id"]>0) & (((base["POINT_ACQUI_ID"]>200) & (base["POINT_ACQUI_ID"]<=600)) | (base["CLASSE_CHOISIE_ID"]=="B"))]
# nbrPointPayeID_B = basePrestID_B["POINT_PAYE_ID"].mean()
# basePrestID_C = base[(base["age_liq_id"]>0) & (((base["POINT_ACQUI_ID"]>600) & (base["POINT_ACQUI_ID"]<=1000)) | (base["CLASSE_CHOISIE_ID"]=="C"))]
# nbrPointPayeID_C = basePrestID_C["POINT_PAYE_ID"].mean()
# proportionClassesID = [len(basePrestID_A), len(basePrestID_B), len(basePrestID_C)]

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

# Densité age d'affiliation
base[(base["PL_ME"] == "PL") & (base["Homme_Femme"] == "H")]["age_1ere_aff"].value_counts().sort_index().to_csv(adresse + f"Tables/affPLH.csv")
base[(base["PL_ME"] == "PL") & (base["Homme_Femme"] == "F")]["age_1ere_aff"].value_counts().sort_index().to_csv(adresse + f"Tables/affPLF.csv")
base[(base["PL_ME"] == "ME") & (base["Homme_Femme"] == "H")]["age_1ere_aff"].value_counts().sort_index().to_csv(adresse + f"Tables/affMEH.csv")
base[(base["PL_ME"] == "ME") & (base["Homme_Femme"] == "F")]["age_1ere_aff"].value_counts().sort_index().to_csv(adresse + f"Tables/affMEF.csv")


