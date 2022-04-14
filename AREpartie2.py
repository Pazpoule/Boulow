import random
import string
import networkx as nx
from math import inf
import os
import imageio
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

import AREpartie1
from AREpartie1 import *



class embarquement:
    def __init__(self, Numero):
        self.numero = Numero
        self.listDebarquementConnecte = []
        self.listPistesEntree = []
        self.listFileAttente =[]
        self.dureeAttenteEstimee = 0
    def printEmbarquement(self):
        print(f"Embarquement {self.numero}")
        print(f"Débarquements connectés : {[debarquement.numero for debarquement in self.listDebarquementConnecte]}")
        print(f"Piste en entrée : {[piste.numero for piste in self.listPistesEntree]}")
        print(f"File d'attente : {[agent.numero for agent in self.listFileAttente]}")
        print(f"Durée d'attente estimée: {self.dureeAttenteEstimee}")
    def evolution(self, temps):
        self.listFileAttente += sum([self.listPistesEntree[indicePiste].listAgentHorsPiste for indicePiste in range(len(self.listPistesEntree))], [])
        debarquement = random.choice(self.listDebarquementConnecte)
        if len(self.listFileAttente) >= int(debarquement.vitesseRemontee):
            [random.choice(debarquement.listPistesDesservies).listAgentAvantPiste.append(self.listFileAttente.pop(0)) for _ in range(int(debarquement.vitesseRemontee))]
        self.dureeAttenteEstimee = len(self.listFileAttente)/np.mean([debarquement.vitesseRemontee for debarquement in self.listDebarquementConnecte])



class debarquement:
    def __init__(self, Numero, ParametreVitesse):
        self.parametreVitesse = ParametreVitesse
        self.numero = Numero
        self.listEmbarquementConnecte = []
        self.listPistesDesservies = []
        self.nombreAgentsParPistesDesservies = 0
        self.nombreCollisionsSurPistesDesservies = 0
        self.vitesseRemontee = ParametreVitesse
    def printDebarquement(self):
        print(f"Débarquement {self.numero}")
        print(f"Embarquements connectés : {[embarquement.numero for embarquement in self.listEmbarquementConnecte]}")
        print(f"Pistes désservies : {[piste.numero for piste in self.listPistesDesservies]}")
        print(f"Nombre d'agents sur pistes desservies : {self.nombreAgentsParPistesDesservies}, Nombre de collisions sur pistes desservies : {self.nombreCollisionsSurPistesDesservies}")
    def evolution(self):
        self.nombreAgentsParPistesDesservies = sum([len(piste.listAgentSurPiste) for piste in self.listPistesDesservies])/len(self.listPistesDesservies)
        self.nombreCollisionsSurPistesDesservies = sum([piste.nombreCollision for piste in self.listPistesDesservies])
        self.vitesseRemontee = self.parametreVitesse / (1+self.nombreAgentsParPistesDesservies)





class stations:
    def __init__(self, NombreRemontees, NombrePistes, NombreAgentsParPiste):
        self.nombreRemontees = NombreRemontees
        self.nombrePistes = NombrePistes
        self.nombreAgentsParPiste = NombreAgentsParPiste
        self.nombreDepartSurPisteInstantane = 5
        self.listEmbarquement = [embarquement(numero) for numero in range(self.nombreRemontees)]
        self.listDebarquement = [debarquement(numero, ParametreVitesse=2*NombreAgentsParPiste) for numero in range(self.nombreRemontees)]
        self.reseau = nx.DiGraph()
        self.reseau.add_nodes_from(self.listEmbarquement)
        self.reseau.add_nodes_from(self.listDebarquement)
        for numero in range(self.nombreRemontees):
            self.listEmbarquement[numero].listDebarquementConnecte.append(self.listDebarquement[numero])
            self.listDebarquement[numero].listEmbarquementConnecte.append(self.listEmbarquement[numero])
            self.reseau.add_edge(self.listEmbarquement[numero], self.listDebarquement[numero])
        self.listPistes = []
        for numero in range(self.nombrePistes):
            debarquementChoisi = random.choices(self.listDebarquement, [math.exp(-3*len(self.reseau[sortie])) for sortie in self.listDebarquement])[0]
            embarquementChoisi = random.choices(self.listEmbarquement, [math.exp(-3*len(list(self.reseau.predecessors(entree)))) for entree in self.listEmbarquement])[0]  #  - 3*((entree, debarquementChoisi) in self.reseau.edges)
            niveauGeneralChoisi = random.randint(0, 100)
            penteChoisie = random.randint(1, 3)
            largeurChoisie = random.randint(30, 40)
            longueurChoisie = random.randint(500, 1000)
            nombreVirageChoisi = random.randint(1, 3)
            largeurMinChoisie = 5
            amplitudeMaxChoisie = 0.8
            etirementMinChoisi = 0.05
            etirementMaxChoisi = 0.1
            dureeCollisionBaseChoisie = 20
            piste = pistes(Numero=numero, ListAgentAvantPiste=[agents(numero, random.choices(["Débutant", "Intermédiaire", "Confirmé", "Expert"], repartitionNiveau(niveauGeneralChoisi))[0]) for numero in range(self.nombreAgentsParPiste)], Pente=penteChoisie, Largeur=largeurChoisie, Longueur=longueurChoisie, NombreVirage=nombreVirageChoisi, LargeurMin=largeurMinChoisie, AmplitudeMax=amplitudeMaxChoisie, EtirementMin=etirementMinChoisi, EtirementMax=etirementMaxChoisi, DureeCollisionBase=dureeCollisionBaseChoisie)
            debarquementChoisi.listPistesDesservies.append(piste)
            embarquementChoisi.listPistesEntree.append(piste)
            self.reseau.add_edge(debarquementChoisi, embarquementChoisi)
            self.listPistes.append(piste)
    def printStation(self):
        print(f"")
    def evolution(self, temps):
        for debarquement in self.listDebarquement:
            debarquement.evolution()
        for embarquement in self.listEmbarquement:
            embarquement.evolution(temps)
        for piste in self.listPistes:
            piste.listAgentHorsPiste = []
            i=0
            while piste.departAgent() and i<self.nombreDepartSurPisteInstantane: i+=1
            piste.evolution()
    def afficheGraphe(self, mouvementEdges, tailleEntree, couleurEntree):
        plt.clf()
        alphaNodes = [1 for _ in range(self.nombreRemontees)] + [0.5 for _ in range(self.nombreRemontees)]
        tailleNodes = [50+2*len(embarquement.listFileAttente) for embarquement in self.listEmbarquement] + [30 for _ in range(self.nombreRemontees)]
        couleurNodes = [((len(embarquement.listFileAttente)/(self.nombreAgentsParPiste*self.nombrePistes))**(1/5), 0, 1-(len(embarquement.listFileAttente)/(self.nombreAgentsParPiste*self.nombrePistes))**(1/5)) for embarquement in self.listEmbarquement] + ["orange" for _ in range(self.nombreRemontees)]
        couleursEdges = ["grey" for _ in range(self.nombreRemontees)] + ["blue" for _ in range(self.nombrePistes)]
        nx.draw_networkx_edges(self.reseau, nx.spring_layout(self.reseau, seed=0), alpha=0.3, edge_color=couleursEdges, connectionstyle='arc3,rad=0.1', style=":", min_source_margin=mouvementEdges)
        nx.draw_networkx_nodes(self.reseau, nx.spring_layout(self.reseau, seed=0), node_size=tailleNodes, alpha=alphaNodes, node_color=couleurNodes) # label=[str(node.numero) for node in reseau.nodes]
        plt.pause(0.0001)

def Experience(nombreRemontees, nombrePistes, nombreAgentsParPiste, dureeExperience, affichage = False):
    if affichage:
        plt.figure(figsize=[16, 9])
    station = stations(NombreRemontees=nombreRemontees, NombrePistes=nombrePistes, NombreAgentsParPiste=nombreAgentsParPiste)
    for temps in range(dureeExperience):
        print(round(temps*100/dureeExperience, 0), "%")
        station.evolution(temps)
        if affichage:
            station.afficheGraphe(temps % 3 + 10, tailleEntree=0, couleurEntree=0)
    return station


exp = Experience(nombreRemontees=10, nombrePistes=15, nombreAgentsParPiste=100, dureeExperience=500, affichage=True)

print("-------------------------------Embarquements----------------------------------------")
for embarquement in exp.listEmbarquement:
    embarquement.printEmbarquement()
print("-------------------------------Pistes------------------------------------------------")
for piste in exp.listPistes:
    piste.printPiste()






# plt.figure(figsize=[16, 9])
# plt.title("Graphe de la station de ski")
# for i in range(100):
#     plt.clf()
#     station.afficheGraphe(i % 3 + 10, 100+i*4, i/100)
#     plt.pause(0.01)
#
#

# station.listEmbarquement[0].printEmbarquement()
# station.listDebarquement[0].printDebarquement()
#
#
# station.afficheGraphe(1 % 3 + 10, 100+1*4, 1/100)

# embarquement = station.listEmbarquement[0]
#
# [embarquement.listPistesEntree[indicePiste].listAgentAvantPiste for indicePiste in range(len(embarquement.listPistesEntree))]
#























































