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


DISTANCE_COLLISION = 0.5

dicoNiveau = {
    "Débutant": {"vitesse": 0.1, "amplitude": 10, "periode": 10, "probabilitéEvitement": 0.4},
    "Intermédiaire": {"vitesse": 0.3, "amplitude": 8, "periode": 10, "probabilitéEvitement": 0.5},
    "Confirmé": {"vitesse": 0.5, "amplitude": 5, "periode": 15, "probabilitéEvitement": 0.6},
    "Expert": {"vitesse": 1, "amplitude": 0.5, "periode": 5, "probabilitéEvitement": 0.7},
}

class agent:
    def __init__(self, Numero, Niveau):
        self.numero = Numero
        self.couleur = random.randint(0, 100)
        self.x = 0
        self.y = 0
        self.positionDepart = 0
        self.piste = None
        self.niveau = Niveau
        self.vitesse = dicoNiveau[Niveau]["vitesse"]
        self.amplitude = dicoNiveau[Niveau]["amplitude"]
        self.periode = dicoNiveau[Niveau]["periode"]
        self.probabiliteEvitement = dicoNiveau[Niveau]["probabilitéEvitement"]
        self.enCollision = False
        self.dureeCollisionRestante = 0
    def depart(self):
        self.positionDepart = random.uniform(self.amplitude, self.piste.largeur - self.amplitude)
        self.y = self.positionDepart
    def avance(self):
        self.x += self.vitesse
        self.y = self.amplitude * math.sin(0.8 * self.x * 2*math.pi/self.periode + self.positionDepart) + self.positionDepart


class piste:
    def __init__(self, ListAgentAvantPiste, Pente = 1, Largeur = 30, Longueur = 500, DureeCollisionBase = 10):
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
        plt.xlim(1, self.longueur)
        plt.ylim(0, self.largeur)
        plt.scatter([agent.x for agent in self.listAgentSurPiste], [agent.y for agent in self.listAgentSurPiste], c = [agent.couleur for agent in self.listAgentSurPiste], alpha=0.3)
        if len(self.listCoordonneesCollision) > 0:
            plt.scatter(np.array(self.listCoordonneesCollision)[:, 0], np.array(self.listCoordonneesCollision)[:, 1], c="red", alpha=0.8)
        plt.pause(0.0001)


plt.figure(figsize=[16, 9])
probabiliteNiveau = [0.4, 0.3, 0.2, 0.1]
piste = piste([agent(numero, random.choices(["Débutant", "Intermédiaire", "Confirmé", "Expert"], probabiliteNiveau)[0]) for numero in range(50)])
for minute in range(5000):
    if minute % 50 == 0:
        piste.departAgent()
    piste.evolution()
    piste.affichePiste()
plt.show()


for agent in piste.listAgentSurPiste:
    print(agent.niveau, agent.couleur, agent.x)














# # test des paramètres
# plt.figure(figsize=[16, 9])
# vitesse = 1
# amplitude = 10
# periode = 10
# x = [i for i in range(500)]
# y = [amplitude * math.sin(vitesse * i * 2*math.pi/periode) for i in range(500)]
# plt.plot(x, y)


































