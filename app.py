#APP FLASK (commande : flask run)
# Partie formulaire non utilisée (uniquement appel à l'API)

from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import Form, validators  
from wtforms.fields import StringField
from wtforms import TextField, BooleanField, PasswordField, TextAreaField, validators
from wtforms.widgets import TextArea
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from toolbox.predict import *
import pandas as pd
import xgboost

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

#formulaire d'appel à l'API (facultatif)
class SimpleForm(Form):
    form_id = TextField('id:', validators=[validators.required()])
    
    @app.route("/", methods=['GET', 'POST'])
    def form():
        form = SimpleForm(request.form)
        print(form.errors)

        if request.method == 'POST':
            form_id=request.form['id']
            print(form_id)
            return(redirect('credit/'+form_id)) 
    
        if form.validate():
            # Save the comment here.
            flash('Vous avez demandé l\'ID : ' + form_id)
            redirect('')
        else:
            flash('Veuillez compléter le champ. ')
    
        return render_template('formulaire_id.html', form=form)


#Load Dataframe
path_df = '../../Notebook & Data/data/custom/train.csv'
dataframe = pd.read_csv(path_df)

@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):

    #récupération id client depuis argument url
    #id_client = request.args.get('id_client', default=1, type=int)
    
    #DEBUG
    #print('id_client : ', id_client)
    #print('shape df ', dataframe.shape)
    
    #calcul prédiction défaut et probabilité de défaut
    prediction, proba = predict_flask(id_client, dataframe)

    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)


#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)