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


@st.cache #mise en cache de la fonction pour exécution unique
def chargement_data(path):
    dataframe = pd.read_csv(path)
    return dataframe

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_explanation(id_input, dataframe, model, sample):
    return interpretation(str(id_input), 
        dataframe, 
        model, 
        sample=sample)

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_ligne_data(id, df):
    return df[df['SK_ID_CURR']==int(id)].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

dataframe = chargement_data(path_df_reduced)
liste_id = dataframe['SK_ID_CURR'].tolist()

#affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
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

    API_url = "http://127.0.0.1:5000/credit/" + id_input

    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        classe_predite = API_data['prediction']
        if classe_predite == 1:
            etat = 'client à risque'
        else:
            etat = 'client peu risqué'
        proba = 1-API_data['proba'] 

        #affichage de la prédiction
        prediction = API_data['proba']
        classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
        classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

    st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")

    #affichage de l'explication du score
    with st.spinner('Chargement des détails de la prédiction...'):
        explanation = chargement_explanation(str(id_input), 
        dataframe, 
        StackedClassifier(), 
        sample=False)
    #st.success('Done!')
    
    #Affichage des graphes    
    graphes_streamlit(explanation)

    st.subheader("Définition des groupes")
    st.markdown("\
    \n\
    * Client : la valeur pour le client considéré\n\
    * Moyenne : valeur moyenne pour l'ensemble des clients\n\
    * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
    * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
    * Similaires : valeur moyenne pour les 20 clients les plus proches du client\
    considéré sur les critères sexe/âge/revenu/durée/montant du crédit\n\n\
    ")

    #Affichage du dataframe d'explicabilité
    #st.write(explanation)

    #Détail des explications
    st.subheader('Traduction des explication')
    chaine_explanation, df_explanation = df_explain(explanation)
    chaine_features = '\n\
    '
    for x, y in zip(df_explanation['Feature'], df_explanation['Nom francais']):
        chaine_features += '* **' + str(x) + ' ** '+str(y) +'\n'\
        ''
    st.markdown(chaine_features)

    #st.write(df_explanation, unsafe_allow_html=True)

    #Modifier le profil client en modifiant une valeur
    #st.subheader('Modifier le profil client')
    st.sidebar.header("Modifier le profil client")
    st.sidebar.markdown('Cette section permet de modifier une des valeurs les plus caractéristiques du client et de recalculer son score')
    features = explanation['feature'].values.tolist()
    liste_features = tuple([''] + features)
    feature_to_update = ''
    feature_to_update = st.sidebar.selectbox('Quelle caractéristique souhaitez vous modifier', liste_features)

    #st.write(dataframe.head())

    if feature_to_update != '':
        value_min = dataframe[feature_to_update].min()
        value_max = dataframe[feature_to_update].max()
        #st.write(list(explanation['feature'].values))
        #st.write(explanation['feature'].values[0])
        default_value = explanation[explanation['feature'] == feature_to_update]['customer_values'].values[0]
        #st.write(default_value)


        min_value = float(dataframe[feature_to_update].min())
        max_value = float(dataframe[feature_to_update].max())

        if (min_value, max_value) == (0,1): 
            step = float(1)
        else :
            step = float((max_value - min_value) / 20)

        update_val = st.sidebar.slider(label = 'Nouvelle valeur (valeur d\'origine : ' + str(default_value)[:4] + ')',
            min_value = min_value,
            max_value =max_value,
            value = default_value,
            step = step)

        if update_val != default_value:
            time.sleep(0.5)
            update_predict, proba_update = predict_update(id_input, dataframe, feature_to_update, update_val)
            if update_predict == 1:
                etat = 'client à risque'
            else:
                etat = 'client peu risqué'
            chaine = 'Nouvelle prédiction : **' + etat +  '** avec **' + str(round((proba_update[0][1])*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'
            st.sidebar.markdown(chaine)

    st.subheader('Informations relatives au client')
    df_client = chargement_ligne_data(id_input, dataframe).T
    df_client['nom_fr'] = [correspondance_feature(feature) for feature in df_client.index]
    st.write(df_client)
        

else: 
    st.write('Identifiant non reconnu')