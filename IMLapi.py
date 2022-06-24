# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:09:39 2022

@author: Sow Abdoul Aziz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:35:30 2022

@author: Sow Abdoul Aziz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:23:42 2022

@author: Sow Abdoul Aziz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:23:32 2022

@author: aa.sow
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:41:39 2022

@author: Sow Abdoul Aziz
"""
#90041049
# Local Adresse :  http://127.0.0.1:5000/credit/IDclient
# adresse distance : https://app-birro.herokuapp.com/credit/idclient
# Github depo : https://github.com/birro90/dash-credit

from flask import Flask
from flask import jsonify
import pandas as pd
import numpy as np
from joblib import load
import os


app = Flask(__name__)

clf = load("RandomForestClassifier.joblib")

sample = pd.read_excel('X_app1.xlsx', index_col='Identifiant de ménage') 
#sample = pd.read_csv('X_app1.csv', index_col='Identifiant de ménage',encoding ='utf-8')

sample.head()

@app.route('/')

def home():

    return jsonify(username='GassMamadou' , email='birro90@hotmail.fr')
    

@app.route('/client/<int:id_client>' , methods=['GET'])

def client(id_client):
    
        id = id_client

        score = clf.predict_proba(sample.loc[[id]])[:,1]

        predict = clf.predict(sample.loc[[id]])

        # round the predict proba value and set to new variable
        percent_score = score*100
               
        id_risk = np.round(percent_score, 3)
        
        # create JSON object
        output = {'prediction': int(predict), 'bon client %': float(id_risk)}

        print('Nouvelle Prédiction : \n', output)
        
        return jsonify(output)
        
if __name__ == '__main__':

    port = os.environ.get("PORT", 5000)
    app.run(host='0.0.0.0', port=port,debug=False)# debug=True