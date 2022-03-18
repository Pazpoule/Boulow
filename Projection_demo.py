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
import Passif.Encode_input_data as inputData

adresse = "C:/Users/pnguyen/Desktop/ROOT/Projection_Photo/"


if __name__ == '__main__':

    PL_ME = "PL"
    debutProj = 2000
    finProj = 2070
    AnneeCalcul = 2022
    NombreNouveauxParAns = 8000
    VA = 42.43
    VS = 2.63
    Rendement = VS / VA

    TGH05 = pd.read_excel(adresse + "Tables/TableMortaliteTGH05.xlsx", index_col=None, sheet_name='ProbaDC')
    TGF05 = pd.read_excel(adresse + "Tables/TableMortaliteTGF05.xlsx", index_col=None, sheet_name='ProbaDC')
    data = inputData.base
    dataCER = data[data["type_ADH"] == "CER"]
    dataDP = data[data["type_ADH"] == "DP"]
    dataDPCER = data[(data["type_ADH"] == "DP") | (data["type_ADH"] == "CER")]
    dataDD = data[data["type_ADH"] == "DD"]
    tauxPL = len(data[data["PL_ME"]=="PL"])/len(data)
    # tauxCER = len(data[data["type_ADH"]=="CER"])/len(data[(data["age_liq_rc"]>0) & (data["age_rad"].isna())]) # taux de CER parmis les liq non rad
    data = data[(data["PL_ME"] == PL_ME) | (data["PL_ME"].isna())]
    data.reset_index(drop=True, inplace=True)
    print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

    tauxIntraAnnuel_RadtoPrest = len(dataDP[dataDP["age_rad"] == dataDP["age_liq_rc"]])/len(dataDP[dataDP["age_rad"]>0])
    tauxIntraAnnuel_RadtoDC = len(dataDP[dataDP["age_rad"] == dataDP["age_dc"]])/len(dataDP[dataDP["age_rad"]>0])
    tauxIntraAnnuel_CERtoPrest = len(dataCER[dataCER["age_rad"] == dataCER["age_liq_rc"]]) / len(dataCER[dataCER["age_liq_rc"] > 0])
    tauxIntraAnnuel_CERtoDC = len(dataCER[dataCER["age_dc"] == dataCER["age_liq_rc"]]) / len(dataCER[dataCER["age_liq_rc"] > 0])
    tauxIntraAnnuel_PresttoDC = len(dataDP[dataDPCER["age_dc"] == dataDPCER["age_liq_rc"]])/len(dataDPCER[(dataDPCER["age_rad"]>0) & (dataDPCER["age_liq_rc"]>0)])


    # Def des points
    data["age_sortie"] = pd.DataFrame({"0": np.select([data["age_liq_rc"] > 0, data["age_liq_rc"].isna()], [data["age_liq_rc"], math.inf]), "1": np.select([data["age_rad"] > 0, data["age_rad"].isna()], [data["age_rad"], math.inf]), "2": np.select([data["age_dc"] > 0, data["age_dc"].isna()], [data["age_dc"], math.inf])}).min(axis=1)
    data["age_derniereCot"] = pd.DataFrame({"0": list(data["age_sortie"]), "1": data["age"]}).min(axis=1)
    data["PointsCotiseParAn"] = data["PTS_RC_CARR"] / np.select([data["age_derniereCot"] - data["age_1ere_aff"]==0, data["age_derniereCot"] - data["age_1ere_aff"]!=0], [1, data["age_derniereCot"] - data["age_1ere_aff"]])
    pointsParAn = pd.DataFrame(data["PointsCotiseParAn"].value_counts())
    pointsParAn.sort_index(inplace=True)
    plt.figure(figsize=[16, 9])
    plt.title("Points")
    sns.distplot(pointsParAn, bins=1000, hist=True, kde=False, kde_kws={'shade': True, 'linewidth': 3}, label="Points")



    

    # ----------------------------------------------------------------------------------- Définition des différentes tables de passage
    plt.figure(figsize=[16, 9])
    plt.title("Tables de passage")

    ageActiftoRad = pd.DataFrame(dataDP["age_rad"].value_counts()) # On exclut les CER radiés apres liq
    ageActiftoRad.sort_index(inplace=True)
    for indice in ageActiftoRad.index:# On définit l'assiette par rapport à un age comme étant tous les adh qui sont plus agé mais qui n'ont ni radie avant ni liq ni décédé, et sont bien affilié, on exclu les personne ni adie ni liq ni decede de l'assiette
        ageActiftoRad.loc[indice, "assiette"] = len(dataDPCER[(dataDPCER["age"] >= indice) & ((dataDPCER["age_rad"].isna()) | (dataDPCER["age_rad"] >= indice)) & ((dataDPCER["age_liq_rc"].isna()) | (dataDPCER["age_liq_rc"] >= indice)) & ((dataDPCER["age_dc"].isna()) | (dataDPCER["age_dc"] >= indice)) & ((dataDPCER["age_1ere_aff"].isna()) | (dataDPCER["age_1ere_aff"] <= indice)) & ((dataDPCER["age_rad"]>0) | (dataDPCER["age_liq_rc"]>0) | (dataDPCER["age_dc"]>0))])
    ageActiftoRad["taux"] = ageActiftoRad["age_rad"] / ageActiftoRad["assiette"]
    ageActiftoRad.sort_index(inplace=True)
    ageActiftoRad.to_csv(adresse + f"Tables/Loi_ActiftoRad{PL_ME}.csv")
    plt.plot(ageActiftoRad["taux"])

    ageActiftoCER = pd.DataFrame(dataCER["age_liq_rc"].value_counts())
    ageActiftoCER.sort_index(inplace=True)
    ageActiftoCER.columns = ["age_liq_rc"]
    for indice in ageActiftoCER.index:
        ageActiftoCER.loc[indice, "assiette"] = len(dataDPCER[(dataDPCER["age"] >= indice) & ((dataDPCER["age_rad"].isna()) | (dataDPCER["age_rad"] >= indice)) & ((dataDPCER["age_liq_rc"].isna()) | (dataDPCER["age_liq_rc"] >= indice)) & ((dataDPCER["age_dc"].isna()) | (dataDPCER["age_dc"] >= indice)) & ((dataDPCER["age_1ere_aff"].isna()) | (dataDPCER["age_1ere_aff"] <= indice)) & ((dataDPCER["age_rad"]>0) | (dataDPCER["age_liq_rc"]>0) | (dataDPCER["age_dc"]>0))])
    ageActiftoCER["taux"] = ageActiftoCER["age_liq_rc"] / ageActiftoCER["assiette"]
    ageActiftoCER.sort_index(inplace=True)
    ageActiftoCER.to_csv(adresse + f"Tables/Loi_ActiftoCER{PL_ME}.csv")
    plt.plot(ageActiftoCER['taux'])
    
    ageCERtoPrest = pd.DataFrame(dataCER["age_rad"].value_counts())
    ageCERtoPrest.sort_index(inplace=True)
    for indice in ageCERtoPrest.index:
        ageCERtoPrest.loc[indice, "assiette"] = len(dataCER[(dataCER["age"] >= indice) & ((dataCER["age_rad"].isna()) | (dataCER["age_rad"] >= indice)) & (dataCER["age_liq_rc"] <= indice) & ((dataCER["age_dc"].isna()) | (dataCER["age_dc"] >= indice)) & ((dataCER["age_1ere_aff"].isna()) | (dataCER["age_1ere_aff"] <= indice)) & ((dataCER["age_rad"]>0) | (dataCER["age_dc"]>0))])
    ageCERtoPrest["taux"] = ageCERtoPrest["age_rad"] / ageCERtoPrest["assiette"]
    ageCERtoPrest.sort_index(inplace=True)
    ageCERtoPrest.to_csv(adresse + f"Tables/Loi_CERtoPrest{PL_ME}.csv")
    plt.plot(ageCERtoPrest["taux"])

    ageRadtoPrest = pd.DataFrame(dataDP.loc[dataDP['age_rad']>0, "age_liq_rc"].value_counts())
    ageRadtoPrest.sort_index(inplace=True)
    for indice in ageRadtoPrest.index:
        ageRadtoPrest.loc[indice, "assiette"] = len(dataDP[(dataDP["age"] >= indice) & (dataDP["age_rad"] <= indice) & ((dataDP["age_liq_rc"].isna()) | (dataDP["age_liq_rc"] >= indice)) & ((dataDP["age_dc"].isna()) | (dataDP["age_dc"] >= indice)) & ((dataDP["age_1ere_aff"].isna()) | (dataDP["age_1ere_aff"] <= indice)) & ((dataDP["age_liq_rc"]>0) | (dataDP["age_dc"]>0))])
    ageRadtoPrest["taux"] = ageRadtoPrest["age_liq_rc"] / ageRadtoPrest["assiette"]
    ageRadtoPrest.sort_index(inplace=True)
    ageRadtoPrest.to_csv(adresse + f"Tables/Loi_RadtoPrest{PL_ME}.csv")
    plt.plot(ageRadtoPrest['taux'])

    ageDDtoPrest = pd.DataFrame(dataDD["age_liq_rc"].value_counts())
    ageDDtoPrest.sort_index(inplace=True)
    for indice in ageDDtoPrest.index:
        ageDDtoPrest.loc[indice, "assiette"] = len(dataDD[(dataDD["age"] >= indice) & ((dataDD["age_liq_rc"].isna()) | (dataDD["age_liq_rc"] >= indice)) & ((dataDD["age_dc"].isna()) | (dataDD["age_dc"] >= indice)) & ((dataDD["age_dc"]>0) | (dataDD["age_liq_rc"] >0))])
    ageDDtoPrest["taux"] = ageDDtoPrest["age_liq_rc"] / ageDDtoPrest["assiette"]
    ageDDtoPrest.sort_index(inplace=True)
    ageDDtoPrest.to_csv(adresse + f"Tables/Loi_DDtoPrest{PL_ME}.csv")
    plt.plot(ageDDtoPrest['taux'])

    ageDC = pd.DataFrame(dataDPCER["age_dc"].value_counts())
    ageDC.sort_index(inplace=True)
    ageDC.columns = ["age_dc"]
    for indice in ageDC.index:
        ageDC.loc[indice, "assiette"] = len(dataDPCER[(dataDPCER["age"] >= indice) & ((dataDPCER["age_dc"].isna()) | (dataDPCER["age_dc"] >= indice)) & ((dataDPCER["age_1ere_aff"].isna()) | (dataDPCER["age_1ere_aff"] <= indice)) & (dataDPCER["age_dc"]>0)])
    ageDC["taux"] = ageDC["age_dc"] / ageDC["assiette"]
    ageDC.sort_index(inplace=True)
    ageDC.to_csv(adresse + f"Tables/Loi_dc{PL_ME}.csv")
    plt.plot(ageDC['taux'])

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
    ageDDtoPrest = lissage(ageDDtoPrest)
    ageDC = lissage(ageDC)



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


    # lissageTaux = True
    # utiliserTableExp = True
    # tauxTable = "taux_lisse" if lissageTaux else "taux"
    # for adh in range(len(data)):
    #     print(round(adh*100/len(data), 2), ' %         Boucle adhérants (décès et liq)')
    #     tableMortalite = TGF05 if data.loc[adh, 'Homme_Femme'] == 'F' else TGH05
    #     tableMortalite = pd.DataFrame({"taux": tableMortalite[data.loc[adh, 'an_nais']], "taux_lisse": tableMortalite[data.loc[adh, 'an_nais']]})
    #     tableDC = ageDC
    #     for age in range(int(data.loc[adh, 'age']), 120):
    #         if random.random() <= tableDC.loc[age, tauxTable]:
    #             data.loc[adh, "age_dc"] = age
    #             data.loc[adh, "Type"] = "DECEDE"
    #             break

    # On échantillonne pour aller plus vite
    frequence = 0.1
    lissageTaux = True
    utiliserTableExp = False
    tauxTable = "taux_lisse" if lissageTaux else "taux"
    dataComplet = data.copy()
    data = data.sample(frac = frequence)
    data.reset_index(drop=True, inplace=True)
    print('--- Echantillonnage: OK --------------------------------------------------------------------------------------------------')


    for adh in range(len(data)):
        print(round(adh*100/len(data), 2), ' %         Boucle adhérants (décès et liq)')
        tableMortalite = TGF05 if data.loc[adh, 'Homme_Femme'] == 'F' else TGH05
        tableMortalite = pd.DataFrame({"taux": tableMortalite[data.loc[adh, 'an_nais']], "taux_lisse": tableMortalite[data.loc[adh, 'an_nais']]})
        tableDC = ageDC if utiliserTableExp else tableMortalite
        for age in range(int(data.loc[adh, 'age']), 120):
            if age != int(data.loc[adh, 'age']) or random.random() <= 0.5: # Suppose que la moitier des adh aujourd'hui ont déjà effectuer leur tirage pour leur age en cours
                tableRad = ageNul if (data.loc[adh, "Type"] in ["PRESTATAIRE", "RADIE", "DECEDE"]) else (ageCERtoPrest if data.loc[adh, "Type"] == "CER" else ageActiftoRad)
                tableLiq = ageNul if (data.loc[adh, "Type"] in ["CER", "PRESTATAIRE", "DECEDE"]) else ((ageRadtoPrest if data.loc[adh, "Type"] == "RADIE" else ageActiftoCER) if data.loc[adh, "type_ADH"] != "DD" else ageDDtoPrest)
                passage = random.choices(["age_rad", "age_liq_rc", "age_dc", "age_encours"], [tableRad.loc[age, tauxTable], tableLiq.loc[age, tauxTable], tableDC.loc[age, tauxTable], 1-tableRad.loc[age, tauxTable]-tableLiq.loc[age, tauxTable]-tableDC.loc[age, tauxTable]])[0]
                data.loc[adh, passage] = age
                if passage == "age_rad" and data.loc[adh, "Type"] == "ACTIF":
                    data.loc[adh, "Type"] = "RADIE"
                    if random.random() <= tauxIntraAnnuel_RadtoPrest:
                        data.loc[adh, "age_liq_rc"] = age
                        data.loc[adh, "Type"] = "PRESTATAIRE"
                    if random.random() <= tauxIntraAnnuel_RadtoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_rad" and data.loc[adh, "Type"] == "CER":
                    data.loc[adh, "Type"] = "PRESTATAIRE"
                    if random.random() <= tauxIntraAnnuel_PresttoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_liq_rc" and data.loc[adh, "Type"] == "RADIE":
                    data.loc[adh, "Type"] = "PRESTATAIRE"
                    if random.random() <= tauxIntraAnnuel_PresttoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_liq_rc" and data.loc[adh, "Type"] == "ACTIF":
                    data.loc[adh, "Type"] = "CER"
                    if random.random() <= tauxIntraAnnuel_CERtoPrest:
                        data.loc[adh, "age_rad"] = age
                        data.loc[adh, "Type"] = "PRESTATAIRE"
                    if random.random() <= tauxIntraAnnuel_CERtoDC:
                        data.loc[adh, "age_dc"] = age
                        data.loc[adh, "Type"] = "DECEDE"
                        break
                if passage == "age_dc":
                    data.loc[adh, "Type"] = "DECEDE"
                    break
    print('--- Projection des décès et liquidations: OK --------------------------------------------------------------------------------------------------')


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
    # todo gérer l'entré de nouveau DD
    baseNewSample = pd.DataFrame({})
    for annee in range(AnneeCalcul, finProj):
        tmpNewSample = data[data["type_ADH"]!="DD"].sample(n = int(NombreNouveauxParAns*frequence)) # On exclut les DD qui n'ont pas de date d'affiliation
        tmpNewSample['an_nais'] = annee - tmpNewSample['age_1ere_aff']
        baseNewSample = baseNewSample.append(tmpNewSample)
    len(baseNewSample)
    baseNewSample['Type'] = 'NEW'
    data = data.append(baseNewSample)
    data.reset_index(drop=True, inplace=True)
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
    data = data.append(dataDecede)
    data["age_sortie"] = pd.DataFrame({"0": np.select([data["age_liq_rc"] > 0, data["age_liq_rc"].isna()], [data["age_liq_rc"], math.inf]), "1": np.select([data["age_rad"] > 0, data["age_rad"].isna()], [data["age_rad"], math.inf]), "2": np.select([data["age_dc"] > 0, data["age_dc"].isna()], [data["age_dc"], math.inf])}).min(axis=1)
    data["PointsAccumule"] = data["PointsCotiseParAn"] * (data["age_sortie"] - data["age_1ere_aff"])

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
        baseProjection.loc[annee - debutProj, 'nbrRadie'] = len(dataDP[(dataDP['age_rad'] <= dataDP["age"]) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrest'] = len(dataDP[(dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDC'] = len(dataDP[dataDP['age_dc'] <= dataDP["age"]]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrCotCER'] = len(dataCER[((dataCER['age_rad'] > dataCER["age"]) | (dataCER['age_rad'].isna())) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrPrestCER'] = len(dataCER[(dataCER['age_rad'] <= dataCER["age"]) & (dataCER['age_liq_rc'] <= dataCER["age"]) & ((dataCER['age_1ere_aff'] <= dataCER["age"]) | (dataCER['age_1ere_aff'].isna())) & ((dataCER['age_dc'] > dataCER["age"]) | (dataCER['age_dc'].isna()))]) / frequence
        baseProjection.loc[annee - debutProj, 'nbrDCCER'] = len(dataCER[dataCER['age_dc'] <= dataCER["age"]]) / frequence

        # todo inclure cer et dd dans proj fi
        baseProjection.loc[annee - debutProj, 'Cotisations'] = sum(dataDP.loc[((dataDP['age_rad'] > dataDP["age"]) | (dataDP['age_rad'].isna())) & ((dataDP['age_liq_rc'] > dataDP["age"]) | (dataDP['age_liq_rc'].isna())) & ((dataDP['age_1ere_aff'] <= dataDP["age"]) | (dataDP['age_1ere_aff'].isna())), "PointsCotiseParAn"])*VA / frequence
        baseProjection.loc[annee - debutProj, 'Prestations'] = sum(dataDP.loc[(dataDP['age_liq_rc'] <= dataDP["age"]) & ((dataDP['age_dc'] > dataDP["age"]) | (dataDP['age_dc'].isna())), "PointsAccumule"])*VS / frequence
        baseProjection.loc[annee - debutProj, 'SoldeTechnique'] = baseProjection.loc[annee - debutProj, 'Cotisations'] - baseProjection.loc[annee - debutProj, 'Prestations']

        # baseProjection.loc[annee - debutProj, 'nbrDDnonLiq'] = len(dataDD[(dataDD['an_liq_rc'] > annee) & ((dataDD['an_1ere_aff'] <= annee) | (dataDD['an_1ere_aff'].isna()))]) * (tauxPL if PL_ME=="PL" else (1-tauxPL)) / frequence
        # baseProjection.loc[annee - debutProj, 'nbrDDPrest'] = len(dataDD[((dataDD['an_liq_rc'] <= annee) | (dataDD['an_liq_rc'].isna())) & ((dataDD['an_dc'] > annee) | (dataDD['an_dc'].isna()))]) * (tauxPL if PL_ME=="PL" else (1-tauxPL)) / frequence
    print('--- Projection par années: OK --------------------------------------------------------------------------------------------------')

    plt.figure(figsize=[16, 9])
    plt.title("Projection démographique")
    plt.plot(baseProjection["Annee"], baseProjection[['nbrCot', 'nbrRadie', 'nbrPrest', 'nbrDC']])
    plt.figure(figsize=[16, 9])
    plt.title("Projection CER")
    plt.plot(baseProjection["Annee"], baseProjection[['nbrCotCER', 'nbrPrestCER', 'nbrDCCER']])
    plt.figure(figsize=[16, 9])
    plt.title("Projection financière")
    plt.plot(baseProjection["Annee"], baseProjection[['Cotisations', 'Prestations', 'SoldeTechnique']])







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






