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

def strToDate(data, variable):
    data[["day", "month", "year"]] = data[variable].str.split("/", expand=True)
    data["day"] = pd.to_numeric(data["day"])
    data["month"] = pd.to_numeric(data["month"])
    data["year"] = pd.to_numeric(data["year"])
    data = data[(data["year"]<=FIN) | (data["year"].isna())]
    data[variable] = pd.to_datetime(data[["day", "month", "year"]])
    data.drop(['day', 'month', 'year'], axis=1, inplace=True)
    return data

def etendre(data, variable):
    for annee in range(DEBUT, FIN + 1):
        data[variable + str(annee)] = 0
        if "DateDebut" in data.columns:
            data["anneeFin"] = pd.to_datetime({"day": 31, "month": 12, "year": data["DateDebut"].dt.year})
            data["anneeDebut"] = pd.to_datetime({"day": 1, "month": 1, "year": data["DateFin"].dt.year})
            data.loc[(annee == data["DateDebut"].dt.year) & ((data["DateFin"] - data["DateDebut"]).dt.days > 0), variable + str(annee)] = data[variable] * ((data[["DateFin", "anneeFin"]].min(axis=1) - data["DateDebut"]).dt.days+1) / ((data["DateFin"] - data["DateDebut"]).dt.days+1)
            data.loc[(annee == data["DateFin"].dt.year) & ((data["DateFin"] - data["DateDebut"]).dt.days > 0), variable + str(annee)] = data[variable] * ((data["DateFin"] - data[["DateDebut", "anneeDebut"]].max(axis=1)).dt.days+1) / ((data["DateFin"] - data["DateDebut"]).dt.days+1)
        else:
            data.loc[annee == data["DateSurvenance"].dt.year, variable + str(annee)] = data[variable]
    data.drop([variable], axis=1, inplace=True)
    return data


adresse = "C:/Users/zpaul/PycharmProjects/Boulow/assets/"
afficheStats = True
DEBUT = 2011
FIN = 2022

importPrime1 = pd.read_csv(adresse + 'PRIME1.csv', sep=';', encoding="ISO-8859-1")[["Police", "Dist.", "Code Pdt", "Date début", "Date fin", "Mt HT", "Commission"]]
importPrime2 = pd.read_csv(adresse + 'PRIME2.csv', sep=';', encoding="ISO-8859-1")[["Police", "Dist.", "Code Pdt", "Date début", "Date fin", "Mt HT", "Commission"]]
importSinistre = pd.read_csv(adresse + 'SINISTRE.csv', sep=';', encoding="ISO-8859-1")[["N° police", "Distributeur", "Produit", "Survenu le", "Charge Total (Ts)"]]
print('--- Import des données: OK --------------------------------------------------------------------------------------------------')

dataPrime = importPrime1.append(importPrime2)
dataPrime.columns = ["Police", "Distributeur", "Produit", "DateDebut", "DateFin", "Prime", "Commission"]
dataPrime["Distributeur"] = pd.to_numeric(dataPrime["Distributeur"])
dataPrime["Produit"] = pd.to_numeric(dataPrime["Produit"])
dataPrime["Prime"] = pd.to_numeric(dataPrime["Prime"].str.replace(',','.'))
dataPrime["Commission"] = pd.to_numeric(dataPrime["Commission"].str.replace(',','.'))
dataPrime = strToDate(dataPrime, "DateFin")
dataPrime = strToDate(dataPrime, "DateDebut")
# On somme les primes par clé en se basant sur la date de fin (car certaine annulation de primes se font en cours d'année et n'ont donc pas la même date de debut) todo merge avec la premiere date de début
dataPrime = dataPrime[["Police", "Distributeur", "Produit", "DateDebut", "DateFin"]].merge(dataPrime.groupby(['Police', 'Distributeur', 'Produit', 'DateFin']).sum().reset_index(), on=["Police", "Distributeur", "Produit", "DateFin"]).drop_duplicates()
dataPrime.sort_values(by=["Police", "Distributeur", "Produit", "DateDebut"], ignore_index=True, inplace=True)
print('--- Formatage PRIME: OK --------------------------------------------------------------------------------------------------')
if afficheStats:
    statDureeCouverture = (dataPrime["DateFin"]-dataPrime["DateDebut"]).value_counts().reset_index()
    plt.figure(figsize=[16, 9])
    plt.title("Durée de couverture en jours")
    plt.hist((dataPrime["DateFin"]-dataPrime["DateDebut"]).dt.days, bins=100)
    plt.figure(figsize=[16, 9])
    plt.title("Distribution des Primes par distributeur")
    statPrimeParDistributeur = dataPrime.groupby("Distributeur").sum()["Prime"].sort_values(ascending=False)
    statPrimeParDistributeur.plot(kind="bar")
    plt.figure(figsize=[16, 9])
    plt.title("Distribution des Primes par produit")
    statPrimeParProduit = dataPrime.groupby("Produit").sum()["Prime"].sort_values(ascending=False)
    statPrimeParProduit.plot(kind="bar")
    print('--- Statistiques PRIME: OK --------------------------------------------------------------------------------------------------')
dataFormatePrime = dataPrime.copy()
dataFormatePrime = etendre(dataFormatePrime, "Prime")
dataFormatePrime = etendre(dataFormatePrime, "Commission")
dataFormatePrime = dataFormatePrime.groupby(["Police", "Distributeur", "Produit"]).sum().reset_index()
dataFormatePrime = dataFormatePrime.fillna(0)


dataSinistre = importSinistre.copy()
dataSinistre["Distributeur"] = pd.to_numeric(dataSinistre["Distributeur"].str.split(" ", expand=True)[0])
dataSinistre["Produit"] = pd.to_numeric(dataSinistre["Produit"].str.split(" ", expand=True)[0])
dataSinistre.columns = ["Police", "Distributeur", "Produit", "DateSurvenance", "Charge"]
dataSinistre = strToDate(dataSinistre, "DateSurvenance")
dataSinistre["Charge"] = pd.to_numeric(dataSinistre["Charge"].str.replace(',','.'))
dataSinistre.drop_duplicates(inplace=True)
dataSinistre.sort_values(by="Police", ignore_index=True, inplace=True)
print('--- Formatage SINISTRE: OK --------------------------------------------------------------------------------------------------')
if afficheStats:
    plt.figure(figsize=[16, 9])
    plt.title("Distribution des Charges par distributeur")
    statChargeParDistributeur = dataSinistre.groupby("Distributeur").sum()["Charge"].sort_values(ascending=False)
    statChargeParDistributeur.plot(kind="bar")
    plt.figure(figsize=[16, 9])
    plt.title("Distribution des Charges par Produit")
    statChargeParProduit = dataSinistre.groupby("Produit").sum()["Charge"].sort_values(ascending=False)
    statChargeParProduit.plot(kind="bar")
    print('--- Statistiques CHARGE: OK --------------------------------------------------------------------------------------------------')
dataFormateSinistre = dataSinistre.copy()
dataFormateSinistre = etendre(dataFormateSinistre, "Charge")
dataFormateSinistre = dataFormateSinistre.groupby(["Police", "Distributeur", "Produit"]).sum().reset_index()
dataFormateSinistre = dataFormateSinistre.fillna(0)


inSinistre_notinPrime = dataFormateSinistre[~dataFormateSinistre["Police"].isin(dataFormatePrime["Police"])]
print(f"Il y a {len(inSinistre_notinPrime)} polices dans SINISTRE qui ne sont pas dans PRIME, soit {round(len(inSinistre_notinPrime)*100/len(dataSinistre), 2)}%")
mauvaisLabel = dataFormateSinistre.merge(dataFormatePrime, on=["Police", "Distributeur", "Produit"], how="outer")
mauvaisLabel = mauvaisLabel[mauvaisLabel["Prime"+str(FIN)].isna()]
print(f"Il y a {len(mauvaisLabel)} polices/Distr/Prod dans SINISTRE qui ne sont pas dans PRIME, soit {round(len(mauvaisLabel)*100/len(dataSinistre), 2)}%")
print('--- Tests Adéquation: OK --------------------------------------------------------------------------------------------------')


# todo récup les exclusions
dataFormate = (dataFormatePrime.merge(dataFormateSinistre, on=["Police", "Distributeur", "Produit"], how="left"))
dataFormate.drop_duplicates(inplace=True)
dataFormate = dataFormate.fillna(0)
dataFormate.to_csv(adresse+"premierjet.csv", sep=";", encoding="latin1")

dataFormateParDistributeur = dataFormate.drop(["Produit", "Police"], axis=1).groupby(["Distributeur"]).sum().reset_index()
for annee in range(DEBUT, FIN+1):
    dataFormateParDistributeur["beneficeBrut"+str(annee)] = dataFormateParDistributeur["Prime"+str(annee)] - dataFormateParDistributeur["Charge"+str(annee)]
    dataFormateParDistributeur["beneficeNet"+str(annee)] = dataFormateParDistributeur["Prime"+str(annee)] - dataFormateParDistributeur["Charge"+str(annee)] - dataFormateParDistributeur["Commission"+str(annee)]
dataFormateParDistributeur["beneficeBrut"] = sum([dataFormateParDistributeur["beneficeBrut"+str(annee)] for annee in range(DEBUT, FIN+1)])
dataFormateParDistributeur["beneficeNet"] = sum([dataFormateParDistributeur["beneficeNet"+str(annee)] for annee in range(DEBUT, FIN+1)])
dataFormateParDistributeur.sort_values(by="beneficeNet2021", inplace=True, ascending=False)
dataFormateParDistributeur.drop(["Prime"+str(annee) for annee in range(DEBUT, FIN+1)]+["Commission"+str(annee) for annee in range(DEBUT, FIN+1)]+["Charge"+str(annee) for annee in range(DEBUT, FIN+1)], axis=1, inplace=True)

dataFormateParProduit = dataFormate.drop(["Distributeur", "Police"], axis=1).groupby(["Produit"]).sum().reset_index()
for annee in range(DEBUT, FIN+1):
    dataFormateParProduit["beneficeBrut"+str(annee)] = dataFormateParProduit["Prime"+str(annee)] - dataFormateParProduit["Charge"+str(annee)]
    dataFormateParProduit["beneficeNet"+str(annee)] = dataFormateParProduit["Prime"+str(annee)] - dataFormateParProduit["Charge"+str(annee)] - dataFormateParProduit["Commission"+str(annee)]
dataFormateParProduit["beneficeBrut"] = sum([dataFormateParProduit["beneficeBrut"+str(annee)] for annee in range(DEBUT, FIN+1)])
dataFormateParProduit["beneficeNet"] = sum([dataFormateParProduit["beneficeNet"+str(annee)] for annee in range(DEBUT, FIN+1)])
dataFormateParProduit.sort_values(by="beneficeNet2021", inplace=True, ascending=False)
dataFormateParProduit.drop(["Prime"+str(annee) for annee in range(DEBUT, FIN+1)]+["Commission"+str(annee) for annee in range(DEBUT, FIN+1)]+["Charge"+str(annee) for annee in range(DEBUT, FIN+1)], axis=1, inplace=True)




dataFormateParDistributeur[["Distributeur", "beneficeNet2021"]].groupby(by="Distributeur").sum().sort_values(by="beneficeNet2021", ascending=False).plot(kind="bar")

# todo mettre en trimestriel




































