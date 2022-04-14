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
import os
import imageio


DISTANCE_COLLISION = 0.4
VITESSE_AFFICHAGE = 0.08
EPSILON = 0.15

dicoNiveau = {
    "Débutant": {"vitesse": 1, "amplitude": 7, "periode": 10, "probabilitéEvitement": 0.2},
    "Intermédiaire": {"vitesse": 3, "amplitude": 5, "periode": 10, "probabilitéEvitement": 0.3},
    "Confirmé": {"vitesse": 5, "amplitude": 3, "periode": 15, "probabilitéEvitement": 0.4},
    "Expert": {"vitesse": 10, "amplitude": 0.5, "periode": 5, "probabilitéEvitement": 0.5},
}

def lissage(liste, force):
    force = int(force)
    liste = [liste[0] for _ in range(force)] + liste + [liste[-1] for _ in range(force)]
    return [np.mean([liste[k] for k in range(i-force, i+force+1)]) for i in range(force, len(liste)-force)]

def creationPiste(largeur, longueur, nombreVirage, largeurMin, amplitudeMax = 0.8, etirementMin = 0.05, etirementMax = 0.1):
    while True:
        parametreGauche = {"amplitude": [random.uniform(-amplitudeMax*largeur/math.pi, amplitudeMax*largeur/math.pi) for _ in range(nombreVirage)], "décalageHorizontal": [longueur/(nombreVirage+1) for _ in range(nombreVirage)], "étirement": [random.uniform(etirementMin*longueur, etirementMax*longueur) for _ in range(nombreVirage)], "décalageVertical": [0 for _ in range(nombreVirage)]}
        parametreDroite = {"amplitude": [random.uniform(-amplitudeMax*largeur/math.pi, amplitudeMax*largeur/math.pi) for _ in range(nombreVirage)], "décalageHorizontal": [longueur/(nombreVirage+1) for _ in range(nombreVirage)], "étirement": [random.uniform(etirementMin*longueur, etirementMax*longueur) for _ in range(nombreVirage)], "décalageVertical": [0 for _ in range(nombreVirage)]}
        for virage in range(1, nombreVirage):
            parametreGauche["décalageVertical"][virage] = parametreGauche["amplitude"][virage-1] * math.atan((longueur/nombreVirage - parametreGauche["décalageHorizontal"][virage-1]) / parametreGauche["étirement"][virage-1]) + parametreGauche["décalageVertical"][virage-1] - parametreGauche["amplitude"][virage] * math.atan((- parametreGauche["décalageHorizontal"][virage]) / parametreGauche["étirement"][virage])
            parametreDroite["décalageVertical"][virage] = parametreDroite["amplitude"][virage-1] * math.atan((longueur/nombreVirage - parametreDroite["décalageHorizontal"][virage-1]) / parametreDroite["étirement"][virage-1]) + parametreDroite["décalageVertical"][virage-1] - parametreDroite["amplitude"][virage] * math.atan((- parametreDroite["décalageHorizontal"][virage]) / parametreDroite["étirement"][virage])
        bordGauche = lissage(sum([[parametreGauche["amplitude"][segment] * math.atan((x - parametreGauche["décalageHorizontal"][segment]) / parametreGauche["étirement"][segment]) + parametreGauche["décalageVertical"][segment] for x in range(int(longueur / nombreVirage))] for segment in range(nombreVirage)], []), longueur / 50)
        bordDroite = lissage(sum([[parametreDroite["amplitude"][segment] * math.atan((x - parametreDroite["décalageHorizontal"][segment]) / parametreDroite["étirement"][segment]) + parametreDroite["décalageVertical"][segment] for x in range(int(longueur / nombreVirage))] for segment in range(nombreVirage)], []), longueur / 50)
        bordGauche = [i + largeur - max(bordGauche) for i in bordGauche]
        bordDroite = [i - min(bordDroite) for i in bordDroite]
        for ecart in range(longueur-len(bordGauche)): # correction de l'arrondi int(longueur / nombreVirage) dans la boucle for
            bordGauche.append(bordGauche[-1])
            bordDroite.append(bordDroite[-1])
        if (min(bordGauche) > 0) and (max(bordDroite) < largeur) and (not [1 for x in range(len(bordGauche)) if bordGauche[x]-bordDroite[x] < largeurMin]):
            return bordGauche, bordDroite

def repartitionNiveau(niveauGeneral):
    return np.array([100 - niveauGeneral, (200 - niveauGeneral) / 3, (100 + niveauGeneral) / 3, niveauGeneral]) / sum([100 - niveauGeneral, (200 - niveauGeneral) / 3, (100 + niveauGeneral) / 3, niveauGeneral])


class agents:
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
    def printAgent(self):
        print(f"Agent {self.numero}")
        print(f"Niveau : {self.niveau}")
        print(f"Position : {self.x, self.y, self.trajectoire}")
        print(f"Piste en cours : {self.piste.numero}")
        print("En collision"*self.enCollision + "Pas en collision"*(1-self.enCollision))
        print(f"Durée restante de collision : {self.dureeCollisionRestante}")
    def depart(self):
        self.trajectoire = random.uniform(self.piste.bordDroit[0] + min(self.amplitude, (self.piste.bordGauche[0]-self.piste.bordDroit[0])/2), self.piste.bordGauche[0] - min(self.amplitude, (self.piste.bordGauche[0]-self.piste.bordDroit[0])/2))
        self.y = self.trajectoire
    def avance(self):
        self.x += self.vitesse
        if (self.x < self.piste.longueur):
            proximiteBord = abs(self.trajectoire - self.piste.bordGauche[self.x]) / abs(self.piste.bordDroit[self.x] - self.piste.bordGauche[self.x]) # à 0% à gauche, à 100% à droite
            self.trajectoire = (self.trajectoire + (self.piste.bordGauche[self.x] - self.piste.bordGauche[self.x - self.vitesse])*(1-proximiteBord)+(self.piste.bordDroit[self.x] - self.piste.bordDroit[self.x - self.vitesse])*(proximiteBord))
            self.y = min(self.amplitude, abs(self.trajectoire - self.piste.bordGauche[self.x])-EPSILON, abs(self.trajectoire - self.piste.bordDroit[self.x])-EPSILON) * math.sin(VITESSE_AFFICHAGE * self.x * 2*math.pi/self.periode + self.orientationDepart) + self.trajectoire


class pistes:
    def __init__(self, Numero, ListAgentAvantPiste, Pente = 1, Largeur = 30, Longueur = 500, NombreVirage = 2, LargeurMin = 5, AmplitudeMax = 0.8, EtirementMin = 0.05, EtirementMax = 0.1, DureeCollisionBase = 20):
        self.numero = Numero
        self.listAgentAvantPiste = ListAgentAvantPiste
        for agent in self.listAgentAvantPiste:
            agent.piste = self
        self.listAgentSurPiste = []
        self.listAgentHorsPiste = []
        self.pente = Pente
        self.largeur = Largeur
        self.longueur = Longueur
        self.nombreVirage = NombreVirage
        self.largeurMin = LargeurMin
        self.dureeCollision = DureeCollisionBase * self.pente
        self.nombreCollision = 0
        self.listCoordonneesCollision = []
        self.listMemoireCollision = []
        self.listLargeurInstantaneeCollision = []
        self.bordGauche, self.bordDroit = creationPiste(self.largeur, self.longueur, self.nombreVirage, self.largeurMin, amplitudeMax=AmplitudeMax, etirementMin=EtirementMin, etirementMax=EtirementMax)
    def printPiste(self):
        print(f"Piste {self.numero} :")
        print(f"Agents avant piste : {[agent.numero for agent in self.listAgentAvantPiste]}")
        print(f"Agents sur piste : {[agent.numero for agent in self.listAgentSurPiste]}")
        print(f"Agents hors piste : {[agent.numero for agent in self.listAgentHorsPiste]}")
        print(f"Nombre d'agents sur pistes : {len(self.listAgentSurPiste)}")
        print(f"Pente : {self.pente}, Largeur : {self.largeur}, Longueur : {self.longueur}, Nombre de virages : {self.nombreVirage}, Largeur min : {self.largeurMin}, Durée de collision : {self.dureeCollision}, Nombre de collision : {self.nombreCollision}")
    def evolution(self):
        for agent in self.listAgentSurPiste:
            if agent.dureeCollisionRestante == 0:
                agent.enCollision = False
                agent.avance()
                if agent.x >= self.longueur:
                    self.sortieAgent(agent)
            else:
                agent.dureeCollisionRestante -= 1
        for element in self.listMemoireCollision:
            element["dureeImmuniteRestante"] -= 1
        self.listMemoireCollision = [element for element in self.listMemoireCollision if element["dureeImmuniteRestante"] != 0]
        for indiceAgent1, agent1 in enumerate(self.listAgentSurPiste[:-1]):
            for indiceAgent2, agent2 in enumerate(self.listAgentSurPiste[indiceAgent1+1:]):
                if self.testCollision(agent1, agent2):
                    self.collision(agent1, agent2)
    def testCollision(self, agent1, agent2):
        return ((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2 <= DISTANCE_COLLISION**2) and ([agent1, agent2] not in [element["agents"] for element in self.listMemoireCollision]) and (random.random() > (agent1.probabiliteEvitement + agent2.probabiliteEvitement)/2)
    def collision(self, agent1, agent2):
        self.nombreCollision += 1
        self.listCoordonneesCollision.append([np.mean([agent1.x, agent2.x]), np.mean([agent1.y, agent2.y])])
        self.listMemoireCollision.append({"agents": [agent1, agent2], "dureeImmuniteRestante": 2*self.dureeCollision})
        self.listLargeurInstantaneeCollision.append(self.bordGauche[int(np.mean([agent1.x, agent2.x]))] - self.bordDroit[int(np.mean([agent1.x, agent2.x]))])
        agent1.enCollision = True
        agent2.enCollision = True
        agent1.dureeCollisionRestante = self.dureeCollision
        agent2.dureeCollisionRestante = self.dureeCollision
    def departAgent(self):
        if len(self.listAgentAvantPiste) > 0:
            self.listAgentSurPiste.append(self.listAgentAvantPiste.pop(0))
            self.listAgentSurPiste[-1].depart()
            return True
        return False
    def sortieAgent(self, agent):
        self.listAgentHorsPiste.append(agent)
        self.listAgentSurPiste = [agentPotentiel for agentPotentiel in self.listAgentSurPiste if agentPotentiel != agent]
    def affichePiste(self):
        plt.clf()
        plt.xlim(0, self.longueur)
        plt.ylim(0, self.largeur+1)
        plt.plot(self.bordGauche, color="steelblue")
        plt.plot(self.bordDroit, color="steelblue")
        plt.scatter([agent.x for agent in self.listAgentSurPiste], [agent.y for agent in self.listAgentSurPiste], c = [agent.couleur for agent in self.listAgentSurPiste], alpha=0.3)
        if len(self.listCoordonneesCollision) > 0:
            plt.scatter(np.array(self.listCoordonneesCollision)[:, 0], np.array(self.listCoordonneesCollision)[:, 1], c="red", alpha=0.8)
        plt.pause(0.0001)

def Experience(nombreAgent = 100, dureeExperience = 10000, dureeEntreDepart = 10, probabiliteNiveau = [0.4, 0.3, 0.2, 0.1], Pente = 1, Largeur = 30, Longueur = 500, NombreVirage = 1, LargeurMin = 5, AmplitudeMax = 0.8, EtirementMin = 0.05, EtirementMax = 0.1, DureeCollisionBase = 20, affichage = False):
    if affichage:
        plt.figure(figsize=[16, 9])
    piste = pistes(Numero = 0, ListAgentAvantPiste = [agents(numero, random.choices(["Débutant", "Intermédiaire", "Confirmé", "Expert"], probabiliteNiveau)[0]) for numero in range(nombreAgent)], Pente=Pente, Largeur=Largeur, Longueur=Longueur, NombreVirage=NombreVirage, LargeurMin=LargeurMin, AmplitudeMax=AmplitudeMax, EtirementMin=EtirementMin, EtirementMax=EtirementMax, DureeCollisionBase=DureeCollisionBase)
    for temps in range(dureeExperience):
        if temps % dureeEntreDepart == 0:
            piste.departAgent()
        piste.evolution()
        if len(piste.listAgentSurPiste) == 0 and len(piste.listAgentHorsPiste) > 0:
            break
        if affichage:
            piste.affichePiste()
    return piste


# --------------------------------------------------------------------- Execution ---------------------------------------------------------------------------

afficheExemple = False
afficheSensibilite = False
nombreIterations = 500

exp = Experience(nombreAgent=100, dureeExperience=5000, dureeEntreDepart=10, probabiliteNiveau=[0.4, 0.3, 0.2, 0.1], Pente=1, Largeur=30, Longueur=500, NombreVirage=2, LargeurMin=5, AmplitudeMax=0.8, EtirementMin=0.05, EtirementMax=0.1, DureeCollisionBase=20, affichage=afficheExemple)

if afficheSensibilite:

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction du nombre d'agents")
    plt.plot([nombreAgent for nombreAgent in range(100)], [np.mean([Experience(nombreAgent=nombreAgent).nombreCollision for _ in range(nombreIterations)]) for nombreAgent in range(100)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction de la durée entre chaque départ sur piste")
    plt.plot([dureeEntreDepart for dureeEntreDepart in range(1, 50)], [np.mean([Experience(dureeEntreDepart=dureeEntreDepart).nombreCollision for _ in range(nombreIterations)]) for dureeEntreDepart in range(1, 50)])

    plt.figure(figsize=[16, 9])
    plt.title("Proportion de skieurs par niveau en fonction du paramètre de niveau général")
    plt.plot([i for i in range(100)], [repartitionNiveau(i)[0] for i in range(100)], label = "Débutants")
    plt.plot([i for i in range(100)], [repartitionNiveau(i)[1] for i in range(100)], label = "Intermédiaires")
    plt.plot([i for i in range(100)], [repartitionNiveau(i)[2] for i in range(100)], label = "Confirmés")
    plt.plot([i for i in range(100)], [repartitionNiveau(i)[3] for i in range(100)], label = "Experts")
    plt.legend()

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction du niveau général des skieurs")
    plt.plot([niveauGeneral for niveauGeneral in range(100)], [np.mean([Experience(probabiliteNiveau=repartitionNiveau(niveauGeneral)).nombreCollision for _ in range(nombreIterations)]) for niveauGeneral in range(100)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction de la pente de la piste")
    plt.plot([pente for pente in range(1, 50)], [np.mean([Experience(Pente=pente).nombreCollision for _ in range(nombreIterations)]) for pente in range(1, 50)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction de la largeur globale de la piste")
    plt.plot([largeur for largeur in range(10, 100)], [np.mean([Experience(Largeur=largeur).nombreCollision for _ in range(nombreIterations)]) for largeur in range(10, 100)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction de la longueur de la piste")
    plt.plot([longueur for longueur in range(300, 3000, 100)], [np.mean([Experience(Longueur=longueur).nombreCollision for _ in range(nombreIterations)]) for longueur in range(300, 3000, 100)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction du nombre de virages de la piste")
    plt.plot([nombreVirage for nombreVirage in range(1, 10)], [np.mean([Experience(NombreVirage=nombreVirage).nombreCollision for _ in range(nombreIterations)]) for nombreVirage in range(1, 10)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction de la durée de base d'une collision")
    plt.plot([dureeCollisionBase for dureeCollisionBase in range(100)], [np.mean([Experience(DureeCollisionBase=dureeCollisionBase).nombreCollision for _ in range(nombreIterations)]) for dureeCollisionBase in range(100)])

    plt.figure(figsize=[16, 9])
    plt.title("Nombre de collisions en fonction de la largeur instantanée de la piste")
    listLargeurInstantaneeCollision = sum([Experience().listLargeurInstantaneeCollision for _ in range(nombreIterations*100)], [])
    listLargeurConstatee = []
    for _ in range(nombreIterations*100):
        bordGauche, bordDroit = creationPiste(30, 500, 2, 5)
        listLargeurConstatee += [bordGauche[x]-bordDroit[x] for x in range(500)]
    histLargeurConstate = plt.hist(listLargeurConstatee, bins=25, label="Largeur instantanée de piste créée aléatoirement")[0]
    histLargeurInstantaneCollision = plt.hist(listLargeurInstantaneeCollision, bins=25, label="Nombre de collision par largeur instantanée")[0]
    plt.legend()
    plt.figure(figsize=[16, 9])
    plt.title("Taux de collisions par largeur instantanée de la piste")
    plt.plot([i for i in range(5, 30)], histLargeurInstantaneCollision/histLargeurConstate)


# todo stat coordonnée en x des collisions a plot en histogramme (voir si plus de collision au début)





# # paramétrage droit
# plt.figure(figsize=[16, 9])
# probabiliteNiveau = [0.4, 0.3, 0.2, 0.1]
# piste = piste([agent(numero, random.choices(["Débutant", "Intermédiaire", "Confirmé", "Expert"], probabiliteNiveau)[0]) for numero in range(100)], AmplitudeMax=0)
# for minute in range(5000):
#     if minute % 10 == 0:
#         piste.departAgent()
#     piste.evolution()
#     piste.affichePiste()
# plt.show()




























































