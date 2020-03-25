#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import time
from urllib.request import urlopen
import json

from toolbox.predict import *


#Load Dataframe
path_df = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/data/custom/train.csv'
#df reduced : 10 % du jeu de donnees initial
path_df_reduced = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/data/custom/dataframe_reduced.csv'

@st.cache
def chargement_data(path):
    dataframe = pd.read_csv(path)
    return dataframe

dataframe = chargement_data(path_df_reduced)
liste_id = dataframe['SK_ID_CURR'].tolist()

#affichage formulaire
st.title('Dashboard Scoring Credit')
st.markdown("Prédictions de scoring client et comparaison à l'ensemble des clients")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
#chaine = "l'id Saisi est " + str(id_input)
#st.write(chaine)

sample_en_regle = str(list(dataframe[dataframe['LABELS'] == 0].sample(5)[['SK_ID_CURR', 'LABELS']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(dataframe[dataframe['LABELS'] == 1].sample(5)[['SK_ID_CURR', 'LABELS']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)

elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API
    #Appel de l'API : 
    data_load_state = st.text('Chargement de la prédiction...')
    API_url = "http://127.0.0.1:5000/credit/" + id_input
    json_url = urlopen(API_url)

    API_data = json.loads(json_url.read())
    classe_predite = API_data['prediction']
    if classe_predite == 1:
        etat = 'client en défaut'
        proba = 1-API_data['proba'] 
    else:
        etat = 'client régulier'
        proba = API_data['proba']

    #affichage de la prédiction
    prediction = API_data['proba']
    classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
    classe_reelle = str(classe_reelle).replace('0', 'en règle').replace('1', 'en défaut')
    chaine = 'Prédiction : ' + etat +  ' avec ' + str(round(proba*100)) + '% de probabilité (classe réelle : '+str(classe_reelle) + ')'
    st.write(chaine)
    st.write(' ')#espace

    #affichage de l'explication du score
    explanation = interpretation(str(id_input), 
        dataframe, 
        StackedClassifier(), 
        sample=False)

    st.write(explanation)

    st.write(df_explain(explanation), unsafe_allow_html=True)

    st.write(dataframe[dataframe['SK_ID_CURR']==int(id_input)])

else: 
    st.write('Identifiant non reconnu')