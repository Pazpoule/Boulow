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

# todo sans bord = bord avec amplitude nulle et offset nulle, modif largeur, longueur
# todo l'amplitude des agent doit dépendre de la largeur max de la piste


DISTANCE_COLLISION = 0.5
VITESSE_AFFICHAGE = 0.08

dicoNiveau = {
    "Débutant": {"vitesse": 1, "amplitude": 7, "periode": 10, "probabilitéEvitement": 0.4},
    "Intermédiaire": {"vitesse": 3, "amplitude": 5, "periode": 10, "probabilitéEvitement": 0.5},
    "Confirmé": {"vitesse": 5, "amplitude": 3, "periode": 15, "probabilitéEvitement": 0.6},
    "Expert": {"vitesse": 10, "amplitude": 0.5, "periode": 5, "probabilitéEvitement": 0.7},
}

class agent:
    def __init__(self, Numero, Niveau):
        self.numero = Numero
        self.couleur = random.randint(0, 100)
        self.x = 0
        self.y = 0
        self.trajectoire = 0
        self.orientationDepart = random.uniform(0, 2 * math.pi)
        self.piste = None
        self.niveau = Niveau
        self.vitesse = dicoNiveau[Niveau]["vitesse"]
        self.amplitude = dicoNiveau[Niveau]["amplitude"]
        self.periode = dicoNiveau[Niveau]["periode"]
        self.probabiliteEvitement = dicoNiveau[Niveau]["probabilitéEvitement"]
        self.enCollision = False
        self.dureeCollisionRestante = 0
    def depart(self):
        if self.piste.bord:
            self.trajectoire = random.uniform(self.piste.bordDroit[0] + self.amplitude, self.piste.bordGauche[0] - self.amplitude)
        else:
            self.trajectoire = random.uniform(self.amplitude, self.piste.largeur - self.amplitude)
        self.y = self.trajectoire
    def avance(self):
        self.x += self.vitesse
        if self.piste.bord and (self.x < self.piste.longueur):
            self.trajectoire = (self.trajectoire + self.piste.bordGauche[self.x] - self.piste.bordGauche[self.x - self.vitesse]) if (self.piste.bordGauche[self.x] - self.trajectoire < self.trajectoire - self.piste.bordDroit[self.x]) else (self.trajectoire + self.piste.bordDroit[self.x] - self.piste.bordDroit[self.x - self.vitesse])
        self.y = self.amplitude * math.sin(VITESSE_AFFICHAGE * self.x * 2*math.pi/self.periode + self.orientationDepart) + self.trajectoire


class piste:
    def __init__(self, ListAgentAvantPiste, Bord = False, BordGauche = {"amplitude": 5, "décalageHorizontal": 250, "décalageVertical": 23, "étirement": 50}, BordDroit = {"amplitude": 5, "décalageHorizontal": 250, "décalageVertical": 7, "étirement": 50}, Pente = 1, Largeur = 30, Longueur = 500, DureeCollisionBase = 10):
        self.listAgentAvantPiste = ListAgentAvantPiste
        for agent in self.listAgentAvantPiste:
            agent.piste = self
        self.listAgentSurPiste = []
        self.listAgentHorsPiste = []
        self.pente = Pente
        self.largeur = Largeur
        self.longueur = Longueur
        self.dureeCollision = DureeCollisionBase * self.pente
        self.nombreCollision = 0
        self.listCoordonneesCollision = []
        self.bord = Bord
        self.bordGauche = [BordGauche["amplitude"] * math.atan((i-BordGauche["décalageHorizontal"]) / BordGauche["étirement"]) + BordGauche["décalageVertical"] for i in range(self.longueur)]
        self.bordDroit = [BordDroit["amplitude"] * math.atan((i-BordDroit["décalageHorizontal"]) / BordDroit["étirement"]) + BordDroit["décalageVertical"] for i in range(self.longueur)]
    def evolution(self):
        for agent in self.listAgentSurPiste:
            if agent.dureeCollisionRestante == 0:
                agent.enCollision = False
                agent.avance()
                if agent.x >= self.longueur:
                    self.sortieAgent(agent)
            else:
                agent.dureeCollisionRestante -= 1
        for indiceAgent1, agent1 in enumerate(self.listAgentSurPiste[:-1]):
            for indiceAgent2, agent2 in enumerate(self.listAgentSurPiste[indiceAgent1+1:]):
                if self.testCollision(agent1, agent2):
                    self.collision(agent1, agent2)
    def testCollision(self, agent1, agent2):
        return ((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2 <= DISTANCE_COLLISION**2) and (not agent1.enCollision or not agent2.enCollision) and (random.random() > (agent1.probabiliteEvitement + agent2.probabiliteEvitement)/2)
    def collision(self, agent1, agent2):
        self.nombreCollision += 1
        agent1.enCollision = True
        agent2.enCollision = True
        agent1.dureeCollisionRestante = self.dureeCollision
        agent2.dureeCollisionRestante = self.dureeCollision
        self.listCoordonneesCollision.append([np.mean([agent1.x, agent2.x]), np.mean([agent1.y, agent2.y])])
    def departAgent(self):
        if len(self.listAgentAvantPiste) > 0:
            self.listAgentSurPiste.append(self.listAgentAvantPiste.pop(0))
            self.listAgentSurPiste[-1].depart()
    def sortieAgent(self, agent):
        self.listAgentHorsPiste.append(agent)
        self.listAgentSurPiste = [agentPotentiel for agentPotentiel in self.listAgentSurPiste if agentPotentiel != agent]
    def affichePiste(self):
        plt.clf()
        plt.xlim(0, self.longueur)
        plt.ylim(0, self.largeur)
        if self.bord:
            plt.plot(self.bordGauche, color="steelblue")
            plt.plot(self.bordDroit, color="steelblue")
        plt.scatter([agent.x for agent in self.listAgentSurPiste], [agent.y for agent in self.listAgentSurPiste], c = [agent.couleur for agent in self.listAgentSurPiste], alpha=0.3)
        if len(self.listCoordonneesCollision) > 0:
            plt.scatter(np.array(self.listCoordonneesCollision)[:, 0], np.array(self.listCoordonneesCollision)[:, 1], c="red", alpha=0.8)
        plt.pause(0.0001)


plt.figure(figsize=[16, 9])
probabiliteNiveau = [0.4, 0.3, 0.2, 0.1]
piste = piste([agent(numero, random.choices(["Débutant", "Intermédiaire", "Confirmé", "Expert"], probabiliteNiveau)[0]) for numero in range(100)], Bord = True)
for minute in range(5000):
    if minute % 10 == 0:
        piste.departAgent()
    piste.evolution()
    piste.affichePiste()
plt.show()






































