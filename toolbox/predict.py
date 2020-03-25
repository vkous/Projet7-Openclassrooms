import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

path_explainer = 'D:/Google Drive/Projet 7 - Score Credit/Notebook & Data/models/explainer.obj'
    

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


    model_1 = load_modele(PATH_MODEL_1) #Modele Random Forest Classifier
    model_2 = load_modele(PATH_MODEL_2) #Modele XGBoost
    model_stacking = load_modele(PATH_MODEL_STACKING) 

    X_stacked = pd.DataFrame([model_1.predict_proba(X)[:,0],
                                model_2.predict_proba(X)[:,0]]).T 
    X_stacked = pd.DataFrame(np.hstack([X_stacked, X]))

    prediction = model_stacking.predict(np.array(X_stacked))
    proba = model_stacking.predict_proba(np.array(X_stacked))

    return prediction, proba

def predict_2(ID, dataframe):

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
    print('Classe réelle : {}'.format(class_names[X['LABELS'].values[0]]))
    print('FIN DEBUG\n')
    
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
    #50 plus proches voisins
    #df_map['moyenne_voisins']

    print('Temps calcul données comparatives : ', time.time() - start_time)
    start_time = time.time()
    #return exp
    return df_map.sort_values(by='contribution')


def df_explain(dataframe):

    chaine = '<h2>Principales caractéristiques discriminantes</h2>'
    for feature in dataframe['feature']:
        chaine += '<h3>Caractéristique : '+ str(feature) + '</h3><ul>'
        chaine += '<li><b>Prospect : </b>'+str(dataframe[dataframe['feature']==feature]['customer_values'].values[0])
        
        chaine_discrim = ' ' + str(dataframe[dataframe['feature']==feature]['signe'].values[0])
        chaine_discrim += ' ' + str(dataframe[dataframe['feature']==feature]['val_lim'].values[0]) + '(seuil de pénalisation)'

        if dataframe[dataframe['feature']==feature]['contribution'].values[0] == 'default' :
            chaine += '<span style=\'color:red\'>' + chaine_discrim + '</span>' 
        else : 
            chaine += '<span style=\'color:green\'>' + chaine_discrim + '</span>' 

        chaine += '</li><li><b>Moyenne : </b>'+str(dataframe[dataframe['feature']==feature]['moy_global'].values[0])+'</li>'
        chaine += '<li><b>Clients réguliers : </b>'+str(dataframe[dataframe['feature']==feature]['moy_en_regle'].values[0])+'</li>'
        chaine += '<li><b>Clients avec défaut: </b>'+str(dataframe[dataframe['feature']==feature]['moy_defaut'].values[0])+'</li>'
        chaine += '</ul>'
    return chaine

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

