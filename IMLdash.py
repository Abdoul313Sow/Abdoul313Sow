# -*- coding: utf-8 -*-

import streamlit as st
import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import urllib
from urllib.request import urlopen
import requests
import configparser
import json
#import plotly.express as px
#from sklearn.cluster import KMeans
from joblib import load
from PIL import Image
import plotly.graph_objects as go

st.title("Service Intégré d’Accueil et d’Orientation du Var")
#st.write( "Modèle d’aide à la décision pour les orientations IML.")

def main() :
    
    @st.cache
    def load_data():
        data = pd.read_excel("NewBase1.xlsx", index_col='Identifiant de ménage')
        sample = pd.read_csv('BASE.csv', index_col='Identifiant de ménage', encoding ='utf-8')
        #data = pd.read_csv('data_app.csv', index_col='Identifiant de ménage', encoding ='utf-8')
        #sample = pd.read_csv('data_app.csv', index_col='Identifiant de ménage', encoding ='utf-8')
        #description = pd.read_csv("features_description.csv", usecols=['Row', 'Description'],  index_col=0, encoding= 'unicode_escape')
        target = data.iloc[:, 0:]
        return data, sample, target


    def load_model():
        '''loading the trained model'''
        model_load = load("RandomForestClassifier.joblib")
        return model_load


    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn


    @st.cache


    def identite_client(data, id):
        data_client = data.loc[[id]]
        data_client['Age'] = data_client['Age'].map({'A1': '<25ans', 'A2': '25-60ans', 'A3':'>60ans'})
        #data_client['Age_A2'] = data_client['Age_A2'].map({0: '0', 1: '25-60ans'})
        #data_client['Age_A3'] = data_client['Age_A3'].map({0: '0', 1: '<60ans'})
        return data_client

   
    @st.cache
    def load_prediction(sample, id, clf):
        score = clf.predict_proba(sample.loc[[id]])[:,1]
        return score


    @st.cache
    def load_kmeans(sample, id, mdl):
        index = sample.loc[id].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].sample(15)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 
    
    #Loading data……
    data, sample, target = load_data()
    id_client = sample.index.values


    #######################################
    # SIDEBAR
    #######################################

    #Title display
    
    html_temp = """
    <div style="background-color: silver; padding:10px; border-radius:12px">
    <h1 style="color: black; text-align:center">Modèle d’aide à la décision pour les orientations IML</h1>
    </div>
    """
    # <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    st.markdown(html_temp, unsafe_allow_html=True)
    
    image_logo = Image.open('logo.PNG')
    st.sidebar.image(image_logo)

    #Customer ID selection
    st.sidebar.header("**Info du ménage**")
    
    #Loading selectbox
    chk_id = st.sidebar.selectbox("Identifiant de ménage", id_client)

    #Loading general info
    #nb_credits, income_annuity_moy_rate, payment_moy_rate, targets = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
   

    #Average income
   
    #AMT CREDIT
    
    
    # Count plot
    fig, ax = plt.subplots(figsize=(5,5))
    sns.countplot(x=data['Cible'])#, order=['No default', 'Default']
    st.sidebar.pyplot(fig)
        

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    
    #Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)
        
    st.header("**Décision orientation**")
    
    # Deployement prediction  : 141 loca et 142 heroku
    #Appel de l'API : 
    API_url = "http://192.168.1.161:5000/client/" + str(chk_id)
    #API_url = "https://app-birro.herokuapp.com/credit/" + str(chk_id)
   
    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        prediction = API_data['bon client %']
    
    ## credit decision limit
    #253159
    if prediction < 50:
        color = "red"
        message = "demande refusée"
    else:
        color = "green"
        message = "demande acceptée"  
    # gauge
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prediction,
        mode = "gauge+number",
        title = {'text': message},
        delta = {'reference': 100},
        gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 50], 'color': "red"},
                 {'range': [50, 100], 'color': "green"}],
             'bar': {'color': "gray"},
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 1, 'value': 50}}))

    fig.update_layout(font = {'color': color, 'family': "Arial"})
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

    st.markdown("---")
    #Customer information display : Customer Gender, Age …
    st.header("**Information du ménage**")

    if st.checkbox("Voir l'information sur le ménage?"):
                       
        infos_client = identite_client(data, chk_id)
        st.write("Customer age1 : ", infos_client['Age'].values[0])
        #st.write("Customer age2 : ", infos_client['Age_A2'].values[0])
        #st.write("Customer age3 : ", infos_client['Age_A3'].values[0])
        
        
        #st.write("Customer Age : {:.0f} ans".format(int((infos_client["DAYS_BIRTH"]/365)*(-1))))
                 

       
    
 
   
    ###########################################################################"""
    st.markdown("<u>Données du ménage :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))        

        

if __name__ == '__main__':
    main()


