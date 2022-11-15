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

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"
data = pd.read_csv(adresse + 'Tables/DATA_nettoye.csv')
AnneeCalcul = 2021
data = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
data.reset_index(drop=True, inplace=True)

def plotDistrib(dico, normalize=True, titre="Distribution"):
    plt.figure(figsize=[16, 9])
    plt.title(titre)
    for index in dico:
        distrib = pd.DataFrame(dico[index].value_counts())
        if normalize: distrib = distrib / distrib.sum()
        distrib.sort_index(inplace=True)
        plt.plot(distrib, label=str(index))
        plt.legend()

def encode_data_NN(base, AnneeCalcul=AnneeCalcul):
    print("Execute encode_data_NN ---------------------")
    # base = base[(base["type_ADH"] == "DP") | (base["type_ADH"] == "CER")]
    oneHot_profession = pd.get_dummies(base.profession, prefix='profession')
    oneHot_cp = pd.get_dummies(base.cp, prefix='lieu')
    oneHot_statut = pd.get_dummies(base.Statut_ADH, prefix='statut')
    dicoMoyenne = {}
    dicoSD = {}
    for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "age", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        dicoMoyenne[colonne] = base[colonne].mean()
        dicoSD[colonne] = base[colonne].std()
    base_NN = base[["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "age", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]]
    base_NN["bool_liq_rc"] = 1
    base_NN.loc[base_NN["age_liq_rc"].isna(), "bool_liq_rc"] = 0
    base_NN["bool_rad"] = 1
    base_NN.loc[base_NN["age_rad"].isna(), "bool_rad"] = 0
    base_NN["bool_dc"] = 1
    base_NN.loc[base_NN["age_dc"].isna(), "bool_dc"] = 0
    base_NN["age_liq_rc"] = np.select([base_NN["age_liq_rc"]>0, (base_NN["age_liq_rc"].isna()) & (base_NN["age_dc"].isna()), (base_NN["age_liq_rc"].isna()) & (base_NN["age_dc"]>0)], [base_NN["age_liq_rc"], AnneeCalcul - base_NN["an_nais"], base_NN["age_dc"]])
    base_NN["age_rad"] = np.select([base_NN["age_rad"]>0, (base_NN["age_rad"].isna()) & (base_NN["age_dc"].isna()), (base_NN["age_rad"].isna()) & (base_NN["age_dc"]>0)], [base_NN["age_rad"], AnneeCalcul - base_NN["an_nais"], base_NN["age_dc"]])
    base_NN.loc[base_NN["age_dc"].isna(), "age_dc"] = AnneeCalcul - base_NN["an_nais"]
    base_NN["Sexe"] = 0
    base_NN.loc[base["Homme_Femme"] == "F", "Sexe"] = 1
    base_NN["PL"] = 0
    base_NN.loc[base["PL_ME"] == "PL", "PL"] = 1
    base_NN = base_NN.join(oneHot_profession.join(oneHot_cp.join(oneHot_statut)))
    base_NN["BOOL_VERSEMENT_UNIQUE"] = base["BOOL_VERSEMENT_UNIQUE"].astype('int64')
    for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        base_NN[colonne] = (base_NN[colonne] - dicoMoyenne[colonne]) / dicoSD[colonne]
    base_NN["Validation"] = 0
    base_NN.loc[(base_NN["bool_rad"]==1) & (base_NN["bool_liq_rc"]==1) & (base_NN["bool_dc"]==1) & (base_NN.index % 50 == 0), "Validation"] = 1
    base_NN.reset_index(drop=True, inplace=True)
    print("FIN encode_data_NN ---------------------")
    return base_NN, dicoMoyenne, dicoSD

def encode_data_NN_sans_age(base, AnneeCalcul=AnneeCalcul):
    print("Execute encode_data_NN ---------------------")
    # base = base[(base["type_ADH"] == "DP") | (base["type_ADH"] == "CER")]
    base = base.drop('age', axis=1)
    oneHot_profession = pd.get_dummies(base.profession, prefix='profession')
    oneHot_cp = pd.get_dummies(base.cp, prefix='lieu')
    oneHot_statut = pd.get_dummies(base.Statut_ADH, prefix='statut')
    dicoMoyenne = {}
    dicoSD = {}
    for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        dicoMoyenne[colonne] = base[colonne].mean()
        dicoSD[colonne] = base[colonne].std()
    base_NN = base[["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]]
    base_NN["bool_liq_rc"] = 1
    base_NN.loc[base_NN["age_liq_rc"].isna(), "bool_liq_rc"] = 0
    base_NN["bool_rad"] = 1
    base_NN.loc[base_NN["age_rad"].isna(), "bool_rad"] = 0
    base_NN["bool_dc"] = 1
    base_NN.loc[base_NN["age_dc"].isna(), "bool_dc"] = 0
    base_NN["age_liq_rc"] = np.select([base_NN["age_liq_rc"]>0, (base_NN["age_liq_rc"].isna()) & (base_NN["age_dc"].isna()), (base_NN["age_liq_rc"].isna()) & (base_NN["age_dc"]>0)], [base_NN["age_liq_rc"], AnneeCalcul - base_NN["an_nais"], base_NN["age_dc"]])
    base_NN["age_rad"] = np.select([base_NN["age_rad"]>0, (base_NN["age_rad"].isna()) & (base_NN["age_dc"].isna()), (base_NN["age_rad"].isna()) & (base_NN["age_dc"]>0)], [base_NN["age_rad"], AnneeCalcul - base_NN["an_nais"], base_NN["age_dc"]])
    base_NN.loc[base_NN["age_dc"].isna(), "age_dc"] = AnneeCalcul - base_NN["an_nais"]
    base_NN["Sexe"] = 0
    base_NN.loc[base["Homme_Femme"] == "F", "Sexe"] = 1
    base_NN["PL"] = 0
    base_NN.loc[base["PL_ME"] == "PL", "PL"] = 1
    base_NN = base_NN.join(oneHot_profession.join(oneHot_cp.join(oneHot_statut)))
    base_NN["BOOL_VERSEMENT_UNIQUE"] = base["BOOL_VERSEMENT_UNIQUE"].astype('int64')
    for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
        base_NN[colonne] = (base_NN[colonne] - dicoMoyenne[colonne]) / dicoSD[colonne]
    base_NN["Validation"] = 0
    base_NN.loc[(base_NN["bool_rad"]==1) & (base_NN["bool_liq_rc"]==1) & (base_NN["bool_dc"]==1) & (base_NN.index % 50 == 0), "Validation"] = 1
    base_NN.reset_index(drop=True, inplace=True)
    print("FIN encode_data_NN ---------------------")
    return base_NN, dicoMoyenne, dicoSD

base_NN, dicoMoyenne, dicoSD = encode_data_NN_sans_age(data)
# On export ici la base pour la traiter dans le modèle NN
# base_NN.to_csv(adresse + f"Tables/base_NN.csv")

# On import la base résultante du modèle NN
base_NN_projete = pd.read_csv(adresse + 'out.csv')
base_NN_projete.columns = ["age_liq_rc_hat", "age_rad_hat", "age_dc_hat"]
base_NN = data[["NUM_ADHERENT_FONCT", "NUM_ADHERENT", "PointsCotiseParAn"]].join(base_NN.join(base_NN_projete))
# On dénormalise la donnée
for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
    base_NN[colonne] = base_NN[colonne] * dicoSD[colonne] + dicoMoyenne[colonne]
base_NN["age_liq_rc_hat"] = base_NN["age_liq_rc_hat"] * dicoSD["age_liq_rc"] + dicoMoyenne["age_liq_rc"]
base_NN["age_rad_hat"] = base_NN["age_rad_hat"] * dicoSD["age_rad"] + dicoMoyenne["age_rad"]
base_NN["age_dc_hat"] = base_NN["age_dc_hat"] * dicoSD["age_dc"] + dicoMoyenne["age_dc"]
# On applique les seuils de réalisme
base_NN.loc[base_NN["age_liq_rc_hat"]<base_NN["age_1ere_aff"], "age_liq_rc_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_rad_hat"]<base_NN["age_1ere_aff"], "age_rad_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_dc_hat"]<base_NN["age_1ere_aff"], "age_dc_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_liq_rc_hat"]>base_NN["age_dc_hat"], "age_liq_rc_hat"] = base_NN["age_dc_hat"]
base_NN.loc[base_NN["age_rad_hat"]>base_NN["age_dc_hat"], "age_rad_hat"] = base_NN["age_dc_hat"]
# base_NN.loc[base_NN["age_liq_rc_hat"]<60, "age_liq_rc_hat"] = 60
for age in ["age_rad", "age_liq_rc", "age_dc", "ageMinRtetraite", "ageTauxPlein", "age_rad_hat", "age_liq_rc_hat", "age_dc_hat"]:
    base_NN[age] = base_NN[age].round(0)

# plotDistrib({"age_liq_base_NN": base_NN[base_NN["bool_liq_rc"]==1]["age_liq_rc_hat"], "age_liq_data": data["age_liq_rc"]})
# len(base_NN[base_NN["age_liq_rc_hat"]<60])/len(base_NN)
# len(data[data["age_liq_rc"]<60])/len(data)
# len(data[(data["age_liq_rc"]<data["ageMinRtetraite"])])/len(data[data["age_liq_rc"]>0])
# len(base_NN[(base_NN["age_liq_rc_hat"]<base_NN["ageMinRtetraite"]) & (base_NN["bool_liq_rc"]==0)])/len(base_NN[base_NN["bool_liq_rc"]==0])
# plotDistrib({"age_liq<ageMin": base_NN[base_NN["age_liq_rc"]<base_NN["ageMinRtetraite"]], "age_liq_hat_ageMin": base_NN[base_NN["age_liq_rc_hat"]<base_NN["ageMinRtetraite"]]}) # majoitairement ME
# test = data[(data["age_liq_rc"]<data["ageMinRtetraite"])]
# test = base_NN[(base_NN["age_liq_rc_hat"]<base_NN["ageMinRtetraite"]) & (base_NN["bool_liq_rc"]==1)]
# test = base_NN[base_NN["age_liq_rc_hat"]<60]    # anomalie
# plotDistrib({"age_liq_base_NN": base_NN["an_nais"], "test": test["an_nais"]}) # les jeunes
# plotDistrib({"age_liq_base_NN": base_NN["PL"], "test": test["PL"]}) # majoitairement ME
# # todo age min forcé
# # todo changer de base de validation



base_NN.to_csv(adresse + f"Tables/base_NN_projete.csv", index=False)


# On reformate la donnée
base_NN = pd.read_csv(adresse + 'Tables/base_NN_projete.csv')
base_NN["PL_ME"] = np.select([(base_NN["PL"] == 1), (base_NN["PL"] == 0)], ["PL", "ME"])
base_NN["Homme_Femme"] = np.select([(base_NN["Sexe"] == 1), (base_NN["Sexe"] == 0)], ["F", "H"])
base_NN["age"] = AnneeCalcul - base_NN["an_nais"]
testVisualisationColonnes = pd.DataFrame(base_NN.columns)
base_NN["profession"] = base_NN[base_NN.columns[18:106]].idxmax(axis=1).str.replace("profession_", "")
base_NN["cp"] = base_NN[base_NN.columns[106:207]].idxmax(axis=1).str.replace("lieu_", "")
base_NN["Statut_ADH"] = base_NN[base_NN.columns[207:212]].idxmax(axis=1).str.replace("statut_", "")
base_NN["type_ADH"] = np.select([(base_NN["age_liq_rc"] < base_NN["age_rad"]), (base_NN["age_liq_rc"] >= base_NN["age_rad"])], ["CER", "DP"])
base_NN.loc[base_NN["Statut_ADH"] == "CER", "type_ADH"] = "CER"
base_NN.drop(base_NN.columns[18:212], inplace=True, axis=1)

base_NN.to_csv(adresse + f"Tables/base_NN_projete.csv", index=False)


# # On créer le dataset de validation et on récupère les identifiants adhérents
# datasetValidation = base_NN[base_NN["Validation"]==1]
# datasetValidation.reset_index(drop=True, inplace=True)
# for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
#     datasetValidation[colonne] = datasetValidation[colonne].round(0)
# data = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
# data["age_liq_rc"] = np.select([data["age_liq_rc"]>0, (data["age_liq_rc"].isna()) & (data["age_dc"].isna()), (data["age_liq_rc"].isna()) & (data["age_dc"]>0)], [data["age_liq_rc"], AnneeCalcul - data["an_nais"], data["age_dc"]])
# data["age_rad"] = np.select([data["age_rad"]>0, (data["age_rad"].isna()) & (data["age_dc"].isna()), (data["age_rad"].isna()) & (data["age_dc"]>0)], [data["age_rad"], AnneeCalcul - data["an_nais"], data["age_dc"]])
# data.loc[data["age_dc"].isna(), "age_dc"] = AnneeCalcul - data["an_nais"]
# for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
#     data[colonne] = data[colonne].round(0)
#
# datasetValidation = datasetValidation.merge(data[["PL_ME", "Homme_Femme", "type_ADH", "profession", "cp", "Statut_ADH", "an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein", "NUM_ADHERENT_FONCT", "NUM_ADHERENT"]], how="inner", on=["PL_ME", "Homme_Femme", "type_ADH", "profession", "cp", "Statut_ADH", "an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]).drop_duplicates(ignore_index=True)
# datasetValidation.to_csv(adresse + f"Tables/datasetValidation.csv", index=False)




























