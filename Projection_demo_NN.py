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
data = pd.read_csv(adresse + 'Tables/DATA_nettoye.csv')
AnneeCalcul = 2021

def encode_data_NN(base, AnneeCalcul):
    print("Execute encode_data_NN ---------------------")
    base = base[(base["type_ADH"] == "DP") | (base["type_ADH"] == "CER")]
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
    print("FIN encode_data_NN ---------------------")
    return base_NN, dicoMoyenne, dicoSD

base_NN, dicoMoyenne, dicoSD = encode_data_NN(data)
# On export ici la base pour la traiter dans le mod??le NN
# base_NN.to_csv(adresse + f"Tables/base_NN.csv")

# On import la base r??sultante du mod??le NN
base_NN = pd.read_csv(adresse + 'out.csv')
# On d??normalise la donn??e
for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
    base_NN[colonne] = base_NN[colonne] * dicoSD[colonne] + dicoMoyenne[colonne]
base_NN["age_liq_rc_hat"] = base_NN["age_liq_rc_hat"] * dicoSD["age_liq_rc"] + dicoMoyenne["age_liq_rc"]
base_NN["age_rad_hat"] = base_NN["age_rad_hat"] * dicoSD["age_rad"] + dicoMoyenne["age_rad"]
base_NN["age_dc_hat"] = base_NN["age_dc_hat"] * dicoSD["age_dc"] + dicoMoyenne["age_dc"]
# On applique les seuils de r??alisme
base_NN.loc[base_NN["age_liq_rc_hat"]<base_NN["age_1ere_aff"], "age_liq_rc_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_rad_hat"]<base_NN["age_1ere_aff"], "age_rad_hat"] = base_NN["age_1ere_aff"]
base_NN.loc[base_NN["age_dc_hat"]<base_NN["age_1ere_aff"], "age_dc_hat"] = base_NN["age_1ere_aff"]

base_NN.to_csv(adresse + f"Tables/base_NN_projete.csv", index=False)

# On reformate la donn??e
base_NN = pd.read_csv(adresse + 'Tables/base_NN_projete.csv')
base_NN["PL_ME"] = np.select([(base_NN["PL"] == 1), (base_NN["PL"] == 0)], ["PL", "ME"])
base_NN["Homme_Femme"] = np.select([(base_NN["Sexe"] == 1), (base_NN["Sexe"] == 0)], ["F", "H"])
base_NN["type_ADH"] = np.select([(base_NN["age_liq_rc"] < base_NN["age_rad"]), (base_NN["age_liq_rc"] >= base_NN["age_rad"])], ["CER", "DP"])
base_NN["age"] = AnneeCalcul - base_NN["an_nais"]
base_NN["profession"] = base_NN[base_NN.columns[15:15+89]].idxmax(axis=1).str.replace("profession_", "")
base_NN["cp"] = base_NN[base_NN.columns[105:105+101]].idxmax(axis=1).str.replace("lieu_", "")
base_NN["Statut_ADH"] = base_NN[base_NN.columns[207:207+5]].idxmax(axis=1).str.replace("statut_", "")
base_NN.drop(base_NN.columns[13:210], inplace=True, axis=1)

base_NN.to_csv(adresse + f"Tables/base_NN_projete.csv", index=False)

# On cr??er le dataset de validation et on r??cup??re les identifiants adh??rents
datasetValidation = base_NN[base_NN["Validation"]==1]
datasetValidation.reset_index(drop=True, inplace=True)
for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
    datasetValidation[colonne] = datasetValidation[colonne].round(0)
data = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
data["age_liq_rc"] = np.select([data["age_liq_rc"]>0, (data["age_liq_rc"].isna()) & (data["age_dc"].isna()), (data["age_liq_rc"].isna()) & (data["age_dc"]>0)], [data["age_liq_rc"], AnneeCalcul - data["an_nais"], data["age_dc"]])
data["age_rad"] = np.select([data["age_rad"]>0, (data["age_rad"].isna()) & (data["age_dc"].isna()), (data["age_rad"].isna()) & (data["age_dc"]>0)], [data["age_rad"], AnneeCalcul - data["an_nais"], data["age_dc"]])
data.loc[data["age_dc"].isna(), "age_dc"] = AnneeCalcul - data["an_nais"]
for colonne in ["an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]:
    data[colonne] = data[colonne].round(0)

datasetValidation = datasetValidation.merge(data[["PL_ME", "Homme_Femme", "type_ADH", "profession", "cp", "Statut_ADH", "an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein", "NUM_ADHERENT_FONCT", "NUM_ADHERENT"]], how="inner", on=["PL_ME", "Homme_Femme", "type_ADH", "profession", "cp", "Statut_ADH", "an_nais", "age_1ere_aff", "age_liq_rc", "age_rad", "age_dc", "NBR_TRIM_CARR", "PTS_RB_CARR", "PTS_RC_CARR", "ageMinRtetraite", "ageTauxPlein"]).drop_duplicates(ignore_index=True)
datasetValidation.to_csv(adresse + f"Tables/datasetValidation.csv", index=False)

































