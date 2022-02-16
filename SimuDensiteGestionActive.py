# Ce code sert à simuler la gestion activ de facon purement aléatoire
# On fait une marche aléatoire pour un certain nombre de gérants (pile ou face, avec gain et perte en pourcentage) et on regarde quel est le résultat au bout d'une periode


import pandas as pd
import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_table import *
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import webbrowser
import numpy as np
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # On definit le gain moyen et la perte moyenne ainsi que la proba de gain
    gain = 0.05
    perte = -0.05
    probaGainHistorique = 0.5

    # On va afficher le résultat avec dash
    app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Slider(
            id="proba",
            min=-1,
            max=1,
            step=0.01,
            value=probaGainHistorique
        ),
        html.Div(id="probaOutput"),
        html.Br(),
        dcc.Slider(
            id="offsetGain",
            min=-0.01,
            max=0.01,
            step=0.0001,
            value=0.000
        ),
        html.Div(id="gainOutput"),
        html.Br(),
        dcc.Input(
            id='trajets',
            placeholder='Enter a value...',
            value=200
        ),
        dcc.Input(
            id='days',
            placeholder='Enter a value...',
            value=2000
        ),
        html.Br(),
        html.Div(id="moyenne"),
        html.Div(id="mediane"),
        html.Br(),
        html.Div(id="gagnants"),
        html.Div(id="perdants"),
        html.Br(),
        dcc.Graph(id="line-chart", className="lineGraphSimu"),
        dcc.Graph(id="histo", className="lineGraphSimu"),
    ])


    @app.callback(
        [Output("line-chart", "figure"), Output("probaOutput", "children"), Output("gainOutput", "children"), Output("histo", "figure"), Output("moyenne", "children"), Output("mediane", "children"), Output("gagnants", "children"), Output("perdants", "children")],
        [Input("proba", "value"), Input("offsetGain", "value"), Input("days", "value"), Input("trajets", "value")])
    def update_line_chart(proba, offsetGain, days, NbrTrajets):
        # On créer un dataframe qui va contenir nos simulations
        Simu = pd.DataFrame()
        # On def le nombre de trajets, ie le nombre de gérants d'actifs que l'on va simuler
        nombreTrajet = int(NbrTrajets)
        # On récup la proba de gain
        probaGain = proba
        # On fait une iteration par gérants
        for iteration in range(nombreTrajet):
            # Le trajet correspond aux valeurs de sont portefauille a chaque étapes. On commence par une valeur de 100€
            trajet = [100]
            # On def la longeur du trajet
            nombreDeJour = int(days)
            # Pour chaque jour, on fait un tirage aléatoire avec un offset de gain pour avoir une moyenne de gain plutot positive
            for jour in range(1, nombreDeJour + 1):
                trajet.append(trajet[jour - 1] * random.choices([1 + gain + offsetGain, 1 + perte], [probaGain, 1 - probaGain])[0])
            Simu[iteration] = trajet
        # On trace la figure de chaques trajet
        fig = px.line(Simu, x=Simu.index, y=[iteration for iteration in range(nombreTrajet)], log_y=True)
        # On def un dataframe des valeurs d'arrivé trié
        densite = pd.DataFrame({'values': Simu.iloc[len(Simu) - 1]}).sort_values(by='values', ignore_index=True)
        # On compte le nombre de gagnants et de perdants
        pourcentageGagnants = round(len(densite[densite['values'] >  100]) *100 / int(NbrTrajets), 1)
        pourcentagePerdants = round(len(densite[densite['values'] <= 100]) *100 / int(NbrTrajets), 1)
        # On trace l'histogramme des résultats
        figdensite = px.histogram(densite, x="values", marginal="box", nbins=100)

        return fig, proba, offsetGain, figdensite, " Valeur moyenne : "+str(int(densite.mean())), " Valeur mediane  : "+str(int(densite.median())), " Gagnants: "+str(len(densite[densite['values']>100]))+" , "+str(pourcentageGagnants)+" %", " Perdants : "+str(len(densite[densite['values']<=100]))+" , "+str(pourcentagePerdants)+" %"


    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server()


