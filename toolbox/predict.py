import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import lime
import time
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, Booster
from lime import lime_text
import lime.lime_tabular
#import dill

PATH_MODEL_1 = "D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/RFC_Sample_Weights_Total.obj"
#PATH_MODEL_2 = "D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/GradientBoosting_Sample_Weights_Total.obj"
#PATH_MODEL_STACKING = "D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/Stacked_GradientBoosting_Sample_Weights_Total.obj"

PATH_MODEL_2 = "D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/GradientBoosting_Sample_Weights_Total_export_natif.obj"
PATH_MODEL_STACKING = "D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/Stacked_GradientBoosting_Sample_Weights_Total_export_natif.obj"

path_df = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/data/custom/train.csv'
dataframe = pd.read_csv(path_df)
df_columns = dataframe.drop(['Unnamed: 0', 'SK_ID_CURR', 'LABELS'], axis=1).columns.tolist()

path_correspondance_features = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/data/custom/features_correspondance.csv'   
path_explainer = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/explainer.obj'
path_kdtree = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/kdtree.obj'

def load_modele(path):
    '''Renvoie le modele en tant qu\'objet à partir du chemin'''
    if 'GradientBoosting' in str(path):
        #print('Chargement XGBOOST')
        model = XGBClassifier()
        model.load_model(path)
        return model
    else:
        #print('Chargement Pickle')
        return pickle.load(open(path, 'rb'))


def predict(ID, dataframe):
    '''Renvoie la prediction a partir du modele ainsi que les probabilites d\'appartenance à chaque classe'''

    ID = int(ID)
    X = dataframe[dataframe['SK_ID_CURR'] == ID]

    X = X.drop(['Unnamed: 0', 'SK_ID_CURR', 'LABELS'], axis=1)
    print('prediction shape X :  ', X.shape)
    model_1 = load_modele(PATH_MODEL_1) #Modele Random Forest Classifier
    model_2 = load_modele(PATH_MODEL_2) #Modele XGBoost
    model_stacking = load_modele(PATH_MODEL_STACKING) 

    X_stacked = pd.DataFrame([model_1.predict_proba(X)[:,0],
                                model_2.predict_proba(X)[:,0]]).T 
    X_stacked = pd.DataFrame(np.hstack([X_stacked, X]))
    print('prediction shape X_stacked :  ', X_stacked.shape)

    prediction = model_stacking.predict(np.array(X_stacked))
    proba = model_stacking.predict_proba(np.array(X_stacked))

    return prediction, proba


def predict_update(ID, dataframe, feature, value):
    '''Renvoie la prédiction à partir d\'un vecteur X'''
    ID = int(ID)
    X = dataframe[dataframe['SK_ID_CURR'] == ID]
    X[feature] = value
    X = X.drop(['Unnamed: 0', 'SK_ID_CURR', 'LABELS'], axis=1)
    if 'Unnamed: 0.1' in X.columns:
        X = X.drop('Unnamed: 0.1', axis=1)
    proba = predict_function_xgb_stacking(X)
    print(proba)
    print(proba[0])
    print(proba[0][0])
    if proba[0][0] > 0.5:
        return 0, proba
    else:
        return 1, proba


def predict_flask(ID, dataframe):
    '''Fonction de prédiction utilisée par l\'API flask :
    a partir de l'identifiant et du jeu de données
    renvoie la prédiction à partir du modèle'''

    ID = int(ID)
    X = dataframe[dataframe['SK_ID_CURR'] == ID]

    X = X.drop(['Unnamed: 0', 'SK_ID_CURR', 'LABELS'], axis=1)
    proba = predict_function_xgb_stacking(X)
    print(proba)
    print(proba[0])
    print(proba[0][0])
    if proba[0][0] > 0.5:
        return 0, proba
    else:
        return 1, proba


    return predictio, proba

def predict_function_xgb_stacking(X):
    '''Fonction de prédiction utilisée pour Lime
    Prends en entrée le jeu de données à prédire
    Renvoie en sortie les probabilités a posteriori en sortie du modèle'''
    model_1 = load_modele(PATH_MODEL_1) #Modele Random Forest Classifier
    model_2 = load_modele(PATH_MODEL_2) #Modele XGBoost
    model_stacking = load_modele(PATH_MODEL_STACKING) 

    if str(X.shape) == str((243,)):
        predict_model_2 =  model_2.predict_proba(
            pd.DataFrame(X.reshape(1,-1), columns=df_columns))[0]
        #print('\nDEBUG predict_function_xgb_stacking :  predict model 2 shape : ', predict_model_2.shape)
    else:
        predict_model_2 =  model_2.predict_proba(
            pd.DataFrame(X, columns=df_columns))[:,0]
        #print('\nDEBUG predict_function_xgb_stacking :  predict model 2 shape : ', predict_model_2.shape)

    #predict_model_2 = model_2.predict_proba(X)[:,0]

    X_stacked = pd.DataFrame([model_1.predict_proba(X)[:,0],
                                predict_model_2]).T
    X_stacked = pd.DataFrame(np.hstack([X_stacked, X]))

    debug = False
    if debug is True:
        print('\nDEBUG predict_function_xgb_stacking : X_stacked')
        print(X_stacked.shape)
        
    if str(X.shape) == str((243,)):
        return model_stacking.predict_proba(pd.DataFrame(X_stacked.reshape(1,-1), columns=X_stacked.columns))[0]
    else:
        return model_stacking.predict_proba(X_stacked)
            #, columns=X_stacked.columns))

def clean_map(string):
    '''nettoyage des caractères de liste en sortie de LIME as_list'''
    signes = ['=>', '<=', '<', '>']
    for signe in signes :
        if signe in string :
            signe_confirme = signe
        string = string.replace(signe, '____')
    string = string.split('____')
    if string[0][-1] == ' ':
        string[0] = string[0][:-1]

    return (string, signe_confirme)

def interpretation(ID, dataframe, model, sample=False):
    '''Fonction qui fait appel à Lime à partir du modèle de prédiction et du jeu de données'''
    #préparation des données
    print('\n\n\n\n======== Nouvelle Instance d\'explicabilité ========')
    start_time = time.time()
    ID = int(ID)
    class_names = ['OK', 'default']
    dataframe_complet = dataframe.copy()
    if 'Unnamed: 0' in dataframe.columns : 
        dataframe = dataframe.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0.1' in dataframe.columns : 
        dataframe = dataframe.drop('Unnamed: 0.1', axis=1) 
    
    X = dataframe[dataframe['SK_ID_CURR'] == ID]
    
    print('ID client: {}'.format(ID))
    #print('Classe réelle : {}'.format(class_names[X['LABELS'].values[0]]))
    
    print('Temps initialisation : ', time.time() - start_time)
    start_time = time.time()


    #si on souhaite travailler avec un volume réduit de données    
    if sample is True :
        dataframe_reduced = dataframe[dataframe['SK_ID_CURR']==int(ID)]
        dataframe = pd.concat([dataframe_reduced, dataframe.sample(2000, random_state=20)], axis=0)
        del dataframe_reduced

    #fin de préparation des données
    X = X.drop(['SK_ID_CURR', 'LABELS'], axis=1)
    dataframe = dataframe.drop(['SK_ID_CURR', 'LABELS'], axis=1)

    #création de l'objet explainer
    import_explainer = False
    if import_explainer is True:
        print('import explainer true')
        with open(path_explainer, 'rb') as f:
            explainer = dill.load(f)
    else:    
        print('import explainer false')
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = np.array(dataframe.sample(int(0.1*dataframe.shape[0]), random_state=20)),
            feature_names = dataframe.columns,
            training_labels = dataframe.columns.tolist(),
            verbose=1,
            random_state=20,
            mode='classification')

        #with open(path_explainer, 'wb') as f:
        #    dill.dump(explainer, f)


    print('Temps initialisation explainer : ', time.time() - start_time)
    start_time = time.time()

    #explication du modèle pour l'individu souhaité

    exp = explainer.explain_instance(data_row = X.sort_index(axis=1).iloc[0:1,:].to_numpy().ravel(),
        predict_fn = predict_function_xgb_stacking)

    print('Temps instance explainer : ', time.time() - start_time)
    start_time = time.time()

    #traitement des données et comparaison
    fig = exp.as_pyplot_figure()
    #exp.show_in_notebook(text=False)
    df_map = pd.DataFrame(exp.as_list())
    print(df_map)

    df_map['feature'] = df_map[0].apply(lambda x : clean_map(x)[0][0])
    df_map['signe'] = df_map[0].apply(lambda x : clean_map(x)[1])
    df_map['val_lim'] = df_map[0].apply(lambda x: clean_map(x)[0][-1])
    #df_map['ecart'] = df_map[0].apply(lambda x: clean_map(x)[0][-1])
    df_map['ecart'] = df_map[1]

    df_map = df_map[['feature', 'signe', 'val_lim', 'ecart']]
    #global
    df_map['contribution'] = 'normal'
    df_map.loc[df_map['ecart']>=0, 'contribution'] = 'default'
    
    df_map['customer_values'] = [X[feature].mean() for feature in df_map['feature'].values.tolist()]
    df_map['moy_global'] = [dataframe_complet[feature].mean() for feature in df_map['feature'].values.tolist()]
    #clients en règle
    df_map['moy_en_regle'] = [dataframe_complet[dataframe_complet['LABELS'] == 0][feature].mean() for feature in df_map['feature'].values.tolist()]
    #clients en règle
    df_map['moy_defaut'] = [dataframe_complet[dataframe_complet['LABELS'] == 1][feature].mean() for feature in df_map['feature'].values.tolist()]
    #20 plus proches voisins
    index_plus_proches_voisins = nearest_neighbors(X, dataframe_complet, 20)
    df_map['moy_voisins'] = [dataframe_complet[dataframe_complet['Unnamed: 0'].isin(index_plus_proches_voisins)][feature].mean() for feature in df_map['feature'].values.tolist()]

    print('Temps calcul données comparatives : ', time.time() - start_time)
    start_time = time.time()
    df_map = pd.concat([df_map[df_map['contribution'] == 'default'].head(3),
        df_map[df_map['contribution'] == 'normal'].head(3)], axis=0)

    return df_map.sort_values(by='contribution')


def df_explain(dataframe):
    '''Ecrit une chaine de caractéres permettant d\'expliquer l\'influence des features dans le résultat de l\'algorithme '''

    chaine = '##Principales caractéristiques discriminantes##  \n'
    df_correspondance = pd.DataFrame(columns=['Feature','Nom francais'])
    for feature in dataframe['feature']:

        chaine += '### Caractéristique : '+ str(feature) + '('+ correspondance_feature(feature) +')###  \n'
        chaine += '* **Prospect : **'+ str(dataframe[dataframe['feature']==feature]['customer_values'].values[0])
        chaine_discrim = ' (seuil de pénalisation : ' + str(dataframe[dataframe['feature']==feature]['signe'].values[0])
        chaine_discrim +=  str(dataframe[dataframe['feature']==feature]['val_lim'].values[0])

        if dataframe[dataframe['feature']==feature]['contribution'].values[0] == 'default' :
            chaine += '<span style=\'color:red\'>' + chaine_discrim + '</span>  \n' 
        else : 
            chaine += '<span style=\'color:green\'>' + chaine_discrim + '</span>  \n' 

        #chaine += '* **Clients Comparables:**'+str(dataframe[dataframe['feature']==feature]['moy_voisins'].values[0])+ '  \n'
        #chaine += '* **Moyenne Globale:**'+str(dataframe[dataframe['feature']==feature]['moy_global'].values[0])+ '  \n'
        #chaine += '* **Clients réguliers :** '+str(dataframe[dataframe['feature']==feature]['moy_en_regle'].values[0])+ '  \n'
        #chaine += '* ** Clients avec défaut: **'+str(dataframe[dataframe['feature']==feature]['moy_defaut'].values[0])+ '  \n'
        #chaine += ''
        df_correspondance_line = pd.DataFrame(data = np.array([[feature, correspondance_feature(feature)]]), columns = ['Feature', 'Nom francais'])
        #df_correspondance_line = pd.DataFrame(data = {'Feature' : feature, 'Nom francais' : correspondance_feature(feature)})
        df_correspondance = pd.concat([df_correspondance, df_correspondance_line], ignore_index=True)
    return chaine, df_correspondance


def nearest_neighbors(X, dataframe, n_neighbors):
    '''Determine les plus proches voisins de l\'individu X 
    considere a partir d\'un KDTree sur 5 colonnes représentatives de caractéristiques intelligibles
    Renvoie en sortie les indices des k plus proches voisins'''
    cols = ['DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'CODE_GENDER_F', 'CREDIT_TERM', 'CREDIT_INCOME_PERCENT']
    tree = pickle.load(open(path_kdtree, 'rb'))
    dist, ind = tree.query(np.array(X[cols]).reshape(1,-1), k = n_neighbors)
    return ind[0]

def correspondance_feature(feature_name):
    '''A partir du nom d\'une feature, trouve sa correspondance en français'''
    df_correspondance = pd.read_csv(path_correspondance_features)
    df_correspondance['Nom origine'] = df_correspondance['Nom origine'].str[1:]
    try:
        return df_correspondance[df_correspondance['Nom origine'] == feature_name]['Nom français'].values[0]
    except:
        print('correspondance non trouvée')
        return feature_name

def graphes_streamlit(df):
    '''A partir du dataframe, affichage un subplot de 6 graphes représentatif du client comparé à d'autres clients sur 6 features'''
    f, ax = plt.subplots(2, 3, figsize=(10,10), sharex=False)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
    i = 0
    j = 0
    liste_cols = ['Client', 'Moyenne', 'En Règle', 'En défaut','Similaires']
    for feature in df['feature']:

        sns.despine(ax=None, left=True, bottom=True, trim=False)
        sns.barplot(y = df[df['feature']==feature][['customer_values', 'moy_global', 'moy_en_regle', 'moy_defaut', 'moy_voisins']].values[0],
                   x = liste_cols,
                   ax = ax[i, j])
        sns.axes_style("white")

        if len(feature) >= 18:
            chaine = feature[:18]+'\n'+feature[18:]
        else : 
            chaine = feature
        if df[df['feature']==feature]['contribution'].values[0] == 'default':
            chaine += '\n(pénalise le score)'
            ax[i,j].set_facecolor('#ffe3e3') #contribue négativement
            ax[i,j].set_title(chaine, color='#990024')
        else:
            chaine += '\n(améliore le score)'
            ax[i,j].set_facecolor('#e3ffec')
            ax[i,j].set_title(chaine, color='#017320')
            
       
        if j == 2:
            i+=1
            j=0
        else:
            j+=1
        if i == 2:
            break;
    for ax in f.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
    if i!=2: #cas où on a pas assez de features à expliquer (ex : 445260)
        #
        True
    st.pyplot()

    return True


from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
class StackedClassifier(BaseEstimator, ClassifierMixin):
    '''Classe pour implémenter en tant que modèle complet le modèle de stacking '''
    def __init__(self, classifiers=None, meta_classifier=None):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        self.model_1 = load_modele(PATH_MODEL_1)
        self.model_2 = load_modele(PATH_MODEL_2)
        self.model_stacking = load_modele(PATH_MODEL_STACKING) 

        self.X_stacked = pd.DataFrame([self.model_1.predict_proba(np.array(X))[:,0],
                                self.model_2.predict_proba(np.array(X))[:,0]]).T 
        self.X_stacked = pd.DataFrame(np.hstack([X_stacked, np.array(X)]))

        return predict_function_xgb_stacking(X_stacked)

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.meta_classifier.predict(X))

    def predict(self, X):
        return self.meta_classifier.fit_predict(np.array(X))

