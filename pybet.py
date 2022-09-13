import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns
import altair as alt

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow import keras


df = pd.read_csv('atp_data.csv')
df_2018 = pd.read_csv('atp_data_2018.csv')
df_2022 = pd.read_csv('atp_data_2022.csv')

#### Mettre une image 
st.image("datascientest_logo.jpeg")

st.write('''
# PyBet Beat The Bookies :tennis:
''')

st.sidebar.header("PyBet")

genre = st.sidebar.radio("Sélectionnez une partie :",
     ('Le Projet', 'Le Jeu De Données', 'Première Exploration', 'Préparation des données', 'Modélisation', 'Résultat','Prédiction', 'Bilan', 'Pour Aller Plus Loin'))


if genre == 'Le Projet':
     st.write('''
            ### Le Projet
            Ce projet a été réalisé dans le cadre de notre formation en data science via l'organisme [Datascientest](https://datascientest.com/).
            
            L’objectif de ce projet est d’essayer de battre les algorithmes des bookmakers sur l’estimation de la probabilité d’un joueur gagnant un match de tennis et d’obtenir un bon retour sur investissement.
            
            Ce _streamlit_ présente notre démarche pour mener à bien ce projet, depuis l'exploration des données jusqu'à la création des variables explicatives. Les meilleurs résultats que nous avons pu obtenir ne sont pas encore présent dans le streamlit, mais la partie Machine Learning vous permet de tester vous même les variables que nous avons créées sur différents algorithmes.
            
            ''')
        
elif genre == 'Le Jeu De Données':
     st.write('''
            ### Le Jeu De Données
            Nous nous sommes appuyés sur les données du site [kaggle](https://www.kaggle.com/code/edouardthomas/beat-the-bookmakers-with-machine-learning-tennis/notebook). 
            
            Ces données ont été extraites gratuitement depuis le site de [Tennis-Data](http://tennis-data.co.uk/data.php.).
            
            Le jeu contient les données de tous les matchs de tennis joués sur l'ATP World Tour (Grand Chelem, Matsers, Master 1000 & ATP500) depuis janvier 2000, et jusqu'à avril 2022.
                   
            La version initiale "atp_data.csv" qui fait 8.76MB est obtenu directement sur le site kaggle: [Lien vers le jeu de données](https://www.kaggle.com/edouardthomas/atp-matches-dataset). Il est constitué d’une grande collection de données sur tous les matchs de tennis joués entre 2000 et 2018. Ces données sont réparties sur 44708 lignes et 23 colonnes. Chaque ligne représente un match de tennis.
            
            ''')
     if st.checkbox("Afficher les données"):
        st.dataframe(df)  
        


elif genre == 'Première Exploration':
     st.write('''
            ### Première Exploration
            Cette partie est là pour permettre une meilleure appréhension du JDD au travers de quelques visuels.
            ''')
     if st.checkbox("Afficher le descriptif des données du dataframe"):
        st.dataframe(df.describe())
     
     if st.checkbox("Afficher les données manquantes"):
        st.dataframe(df.isna().sum())
        
     if st.checkbox("Afficher le nombre de matchs joués sur chaque surface") :
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.countplot(df['Surface'])
        st.pyplot()
    
     if st.checkbox("Afficher la matrice de corrélation") :
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots(figsize=(15,15))
        sns.heatmap(df.corr(), ax=ax, vmin = -1, vmax = +1, annot = True, cmap = 'coolwarm')
        st.pyplot()

elif genre == 'Préparation des données':
     st.write('''
            ### Nettoyage du JDD
            Le jeu de données contient beaucoup de valeurs manquantes. 
            Les colonnes qui contiennent le plus de valeurs manquantes sont les colonnes représentant les côtes (PSW, PSL, B365W, B365L).            
            D'après la matrice de corrélation, il y a une forte corrélation entre les variables PS et B365. 
            
            Les variables PSW et PSL ont chacune 11965 valeurs manquantes.       
            On a décidé de supprimer ces deux variables et de garder que les variables B365W et B365L.
            
            Ensuite, pour supprimer les valeurs manquantes, on applique la fonction « dropna » pour chaque ligne qui contient des valeurs manquantes.
            De plus, la surface « Carpet » a été supprimée en 2009, on peut alors supprimer les données antérieures à 2010.
            
            ''')
        
     st.write('''
            ### Préparation avancée
            Avant de commencer notre modélisation, on a effectué un réarrangement des variables afin de mieux préparer les données pour une meilleure modélisation. 

            On a changé le nom des variables « winner » et « loser » par « joueur1 » et « joueur2 » en prenant soin de les randomiser.
            Idem pour les autres variables, à chaque fois, on a changé les termes W et L par J1 et J2.
            
            Nous remplaçons ensuite les valeurs de colonne winner par des “1” ou  des “2” en fonction que le gagnant du match soit le joueur1 ou le joueur2. De ce fait, nous nous retrouvons dans un problème de classification binaire.
            
            ''')
        
     st.write('''
            ### Ajout de variables
            Afin d'espérer un meilleur résultat pour notre modèle nous allons ajouter des variables afin de retranscrire certains comportements.            
            ''')
    
     if st.checkbox("Surface_J1 & Surface_J2"):
            st.write('''
            Nous avons remarqué que certains joueurs sont meilleurs sur une surface donnée et donc moins bons sur d’autres surfaces. Ainsi le classement Elo n’est pas suffisant. Nous allons donc créer une variable “surface_J1” et “surface_J2” qui respectivement nous donne le pourcentage de victoire sur la surface du match à venir pour le joueur 1 et 2. Cela va nous permettre de retranscripe la préférence de certain joueur pour des surfaces données.
            ''')
         
     if st.checkbox("ratio_J1 & ratio_J2"):
            st.write('''
            Les performances des joueurs sont conditionnées par leur niveau de forme. Un joueur qui a enchaîné les victoires va potentiellement être plus proche de son meilleur niveau qu’un joueur qui a accumulé les défaites. Nous allons donc faire le ratio entre le nombre de victoires divisé par le nombre de matchs gagnés sur les 30 et 90 derniers jours.
            ''')
       
     if st.checkbox("nb_J1 & nb_J2"):
            st.write('''
            Afin de mieux coller avec le ratio nous ajoutons un variable retournant le nombre de match joué lors des 30 et 90 derniers jours. Cela va nous permettre de pondérer le ratio  précédent et d’intégrer les reprises à la compétition.
            ''')
            
     st.write('''
            On obtient alors un nouveau jeu de données _atp_data_2018_ pour les matchs joués entre 2010 et 2018.            
            ''')
   
     if st.checkbox("Afficher les données du dataset atp_data_2018"):
        st.dataframe(df_2018)

elif genre == 'Modélisation':
     st.write('''
            ### Modélisation
            ''')
     st.write('''
            On utilise un modèle de réseau de neurones.
            ''')
     st.image("artificial-neural-network.png")


     st.write('''
            Modèle séquentielle
            ''')
     st.code("""model = Sequential()
     model.add(Dense(256, input_dim=20, activation='relu'))
     model.add(Dense(128, activation='relu'))
     model.add(Dropout(0.2))
     model.add(Dense(1, activation='sigmoid'))""")
     st.image("model.JPG")
   

     
elif genre == 'Résultat':
     st.write('''
            ### Résultat
            On utilise un modèle de réseau de neurone pré-entrainé.
            ''') 
     
     df= pd.read_csv('atp_data_2022.csv')
     df=df.iloc[df[ df['Year'] == 2015 ].index[0]:,:]
     df.drop(['Loser'], axis=1, inplace=True)
     df=df.reset_index(drop=True)
     df.Winner = df['Winner'] == df['Joueur1']
     df['Winner'] = np.where((df.Winner == True),1,df.Winner)
     df['Winner'] = np.where((df.Winner == 0),2,df.Winner)
     J1 =df.Joueur1.unique()
     J2 =df.Joueur2.unique()
     liste_joueur = list(set().union(J1, J2))
     le = LabelEncoder()
     le.fit(liste_joueur)
     df.Joueur1=le.transform(df.Joueur1)
     df.Joueur2=le.transform(df.Joueur2)
     df.dropna(axis=0, how="any", inplace=True)
     df=df.reset_index(drop=True)
     target = df["Winner"] 
     data = df.drop(["Winner"], axis=1)
     target =  [0 if x==2 else 1 for x in target]
     data = data.drop(["ATP","Location","Tournament","Date","Series","Court","Round","Best of","Comment", "Year", "Month", "Day"], axis=1)
     X_num = data.select_dtypes('number')
     X_cat = data.select_dtypes('object')
     labelencoder = LabelEncoder()
     X_cat=X_cat.apply(LabelEncoder().fit_transform)
     data = pd.concat([X_cat,X_num,], axis=1)
     
     test_size = st.sidebar.slider(label= "Choix de la taille de l'échantillon de train", min_value=0.05, max_value =0.3)
     X_train, X_test, y_train, y_test= train_test_split(data, target,shuffle = False, stratify = None,test_size=test_size)
     X_test_save=X_test
       
     scaler = StandardScaler()  
     scaler.fit(X_train)  
     X_train = scaler.transform(X_train)  
     X_test = scaler.transform(X_test)

     X_test=np.array(X_test)
     X_train=np.array(X_train)
     y_test=np.array(y_test)
     y_train=np.array(y_train)
     model = keras.models.load_model("my_model4.h5")

     pred1 = model.predict(X_test)
     y_pred =  [1 if x> 0.5 else 0 for x in pred1]
     pred2 = np.array([1-i for i in pred1])
     y_proba=np.c_[pred1,pred2]
     column_names = ["Proba_1", "Proba_2"]
     data_proba = pd.DataFrame(y_proba, columns=column_names,index = X_test_save.index.copy())

     df_out = pd.merge(df, data_proba, how = 'left', left_index = True, right_index = True)


     def prediction(x):
       
          if (df_out['Proba_1'][x]) > (df_out['Proba_2'][x]) :
               pred = 1
          else:
               pred = 2
          return pred

     df_out['y_pred'] = pd.Series(df_out.index).apply(prediction)
     df_out["resultat"]= df_out.Winner == df_out.y_pred
     
     df_out["Indice_J1"]=df_out["Cote J1"]*df_out["Proba_1"]
     df_out["Indice_J2"]=df_out["Cote J2"]*df_out["Proba_2"]

     def f_gain(x):
          mise  = 1
   
          if (df_out['resultat'][x] == True) & (df_out['Winner'][x]==1):
               gain = (mise * df_out['Cote J1'][x])-mise
          elif (df_out['resultat'][x] == True) & (df_out['Winner'][x]==2):
               gain = (mise * df_out['Cote J2'][x])-mise
          else:
               gain = -mise
          return gain
     df_out['Gain'] = pd.Series(df_out.index).apply(f_gain)
     
     


     def f_gain_pond(x):
          mise_base =1
   
          if (df_out['resultat'][x] == True) & (df_out['Winner'][x]==1):
               cote =df_out['Cote J1'][x]
               mise = (mise_base)*((df_out['Indice_J1'][x])**2 )
               gain = (mise * cote)-mise
          elif (df_out['resultat'][x] == True) & (df_out['Winner'][x]==2):
               cote =df_out['Cote J2'][x]
               mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
               gain = (mise * cote)-mise
          elif (df_out['resultat'][x] == False) & (df_out['Winner'][x]==1):
               mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
               gain = -mise
          else:    
               mise = (mise_base)*((df_out['Indice_J1'][x])**2 )
               gain = -mise  
          return gain
     
     df_out['Gain_pond'] = pd.Series(df_out.index).apply(f_gain_pond)


     def f_mise_pond(x):
          mise_base =1
   
          if (df_out['resultat'][x] == True) & (df_out['Winner'][x]==1):
               mise = (mise_base)*((df_out['Indice_J1'][x])**2 )
          elif (df_out['resultat'][x] == True) & (df_out['Winner'][x]==2):
               mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
          elif (df_out['resultat'][x] == False) & (df_out['Winner'][x]==1):
               mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
          else:
               mise = (mise_base)*((df_out['Indice_J1'][x])**2 )  
          return mise
     df_out['Mise_pond'] = pd.Series(df_out.index).apply(f_mise_pond)



     df_out.dropna(axis=0, how="any", inplace=True)
     df_out =df_out.reset_index(drop=True)

     df_out.Joueur1=le.inverse_transform(df_out.Joueur1)
     df_out.Joueur2=le.inverse_transform(df_out.Joueur2)


     data_ROI = pd.DataFrame([])

     for i in np.arange(0.8, 1.4, 0.01):
        df_prob = df_out[(((df_out["Indice_J1"] > i) & (df_out["y_pred"] == 1)) | ((df_out["Indice_J2"] > i) & (df_out["y_pred"] == 2))) ]
        gain = df_prob['Gain'].sum()
        nb_paris=df_prob.shape[0]
        ROI = (gain/nb_paris)*100
        
        mise_pond = df_prob['Mise_pond'].sum()
        gain_pond = df_prob['Gain_pond'].sum()
        ROI_pond = (gain_pond/mise_pond)*100
        
        data_ROI = data_ROI.append(pd.DataFrame({'Threshold': i, 'nb_paris': nb_paris, 'Gain' : gain, 'ROI': ROI,'gain_pond':gain_pond,'ROI_pond':ROI_pond }, index=[0]), ignore_index=True)
     if st.checkbox("Afficher tableau ROI"):
          st.dataframe(data_ROI)
     

     indice = st.sidebar.slider(label= "Choix de l'indice de confiance", min_value=0.9, max_value =1.5)
     df_prob = df_out[(((df_out["Indice_J1"] > indice) & (df_out["y_pred"] == 1)) | ((df_out["Indice_J2"] > indice) & (df_out["y_pred"] == 2))) ]

     if st.checkbox("Afficher l'échantillon"):
          st.dataframe(df_prob)

     bankroll = st.sidebar.slider(label= "Choix montant bankroll de départ", min_value=100, max_value =10000)
     bankroll_simple = bankroll
     pourcentage = st.sidebar.slider(label= "Choix pourcentage de bankeroll sur chaque paris", min_value=1, max_value =100)
     p_bankroll = pourcentage/100
     mise = bankroll*p_bankroll
     bank_list = []

     for index, row in df_prob.iterrows():
          mise = bankroll*p_bankroll
          bankroll = (row["Gain"]*mise)+bankroll
          bank_list.append(bankroll)

     
     
     mise = bankroll_simple*p_bankroll
     bank_list_simple = []

     for index, row in df_prob.iterrows():
          bankroll_simple =  (row["Gain"]*mise)+bankroll_simple
          bank_list_simple.append(bankroll_simple)

     df_prob =df_prob.reset_index(drop=True)
     df_prob["Bankroll"]= pd.DataFrame(bank_list)
     df_prob["Bankroll_simple"]= pd.DataFrame(bank_list_simple)

     fig = plt.figure(figsize=(25,10))
     plot = sns.lineplot(data=df_prob, x="Date", y="Bankroll",ci=None)
    
     for index, label in enumerate(plot.get_xticklabels()):
          if index % 25== 0:
               label.set_visible(True)
          else:
               label.set_visible(False)
     if st.checkbox("Afficher le graphique Bankroll (mise proportionelle)"):
          lines = alt.Chart(df_prob, title="Bankroll (mise proportionelle)").mark_area(color="lightblue",line=True).encode(x=alt.X('Date',axis=alt.Axis(labels=False)), y=alt.Y('Bankroll:Q'),tooltip=['Bankroll', 'Date']).interactive()
          st.altair_chart(lines,use_container_width=True)

     fig = plt.figure(figsize=(25,10))
     plot = sns.lineplot(data=df_prob, x="Date", y="Bankroll_simple",ci=None)
    
     for index, label in enumerate(plot.get_xticklabels()):
          if index % 25== 0:
               label.set_visible(True)
          else:
               label.set_visible(False)

     if st.checkbox("Afficher le graphique Bankroll (mise fixe)"):
          lines = alt.Chart(df_prob, title="Bankroll (mise fixe)").mark_area(color="lightblue",line=True).encode(x=alt.X('Date',axis=alt.Axis(labels=False)), y=alt.Y('Bankroll_simple:Q'),tooltip=['Bankroll_simple', 'Date']).interactive()
          st.altair_chart(lines,use_container_width=True)

    

elif genre == 'Prédiction':


       df= pd.read_csv('atp_data_2022.csv')
       df=df.iloc[df[ df['Year'] == 2015 ].index[0]:,:]
       df.drop(['Loser'], axis=1, inplace=True)
       df=df.reset_index(drop=True)
       df.Winner = df['Winner'] == df['Joueur1']
       df['Winner'] = np.where((df.Winner == True),1,df.Winner)
       df['Winner'] = np.where((df.Winner == 0),2,df.Winner)
       J1 =df.Joueur1.unique()
       J2 =df.Joueur2.unique()
       liste_joueur = list(set().union(J1, J2)) 
       le = LabelEncoder()
       le.fit(liste_joueur)
       df.Joueur1=le.transform(df.Joueur1)
       df.Joueur2=le.transform(df.Joueur2)
       df.dropna(axis=0, how="any", inplace=True)
       df=df.reset_index(drop=True)
       target = df["Winner"] 
       data = df.drop(["Winner"], axis=1)
       target =  [0 if x==2 else 1 for x in target]
       data = data.drop(["ATP","Location","Tournament","Date","Series","Court","Round","Best of","Comment", "Year", "Month", "Day"], axis=1)
       X_num = data.select_dtypes('number')
       X_cat = data.select_dtypes('object')
       labelencoder = LabelEncoder()
       X_cat=X_cat.apply(LabelEncoder().fit_transform)
       data = pd.concat([X_cat,X_num,], axis=1)
     
       #test_size = st.sidebar.slider(label= "Choix de la taille de l'échantillon de train", min_value=0.05, max_value =0.3)
       test_size = 0.2
       X_train, X_test, y_train, y_test= train_test_split(data, target,shuffle = False, stratify = None,test_size=test_size)
       X_test_save=X_test
       
       scaler = StandardScaler()  
       scaler.fit(X_train)  
       X_train = scaler.transform(X_train)  
       X_test = scaler.transform(X_test)

       X_test=np.array(X_test)
       X_train=np.array(X_train)
       y_test=np.array(y_test)
       y_train=np.array(y_train)
       model = keras.models.load_model("my_model4.h5")

       pred1 = model.predict(X_test)
       y_pred =  [1 if x> 0.5 else 0 for x in pred1]
       pred2 = np.array([1-i for i in pred1])
       y_proba=np.c_[pred1,pred2]
       column_names = ["Proba_1", "Proba_2"]
       data_proba = pd.DataFrame(y_proba, columns=column_names,index = X_test_save.index.copy())

       df_out = pd.merge(df, data_proba, how = 'left', left_index = True, right_index = True)


       def prediction(x):
       
              if (df_out['Proba_1'][x]) > (df_out['Proba_2'][x]) :
                     pred = 1
              else:
                     pred = 2
              return pred

       df_out['y_pred'] = pd.Series(df_out.index).apply(prediction)
       df_out["resultat"]= df_out.Winner == df_out.y_pred
     
       df_out["Indice_J1"]=df_out["Cote J1"]*df_out["Proba_1"]
       df_out["Indice_J2"]=df_out["Cote J2"]*df_out["Proba_2"]

       def f_gain(x):
              mise  = 1
   
              if (df_out['resultat'][x] == True) & (df_out['Winner'][x]==1):
                     gain = (mise * df_out['Cote J1'][x])-mise
              elif (df_out['resultat'][x] == True) & (df_out['Winner'][x]==2):
                     gain = (mise * df_out['Cote J2'][x])-mise
              else:
                     gain = -mise
              return gain
       df_out['Gain'] = pd.Series(df_out.index).apply(f_gain)
     
     


       def f_gain_pond(x):
              mise_base =1
   
              if (df_out['resultat'][x] == True) & (df_out['Winner'][x]==1):
                     cote =df_out['Cote J1'][x]
                     mise = (mise_base)*((df_out['Indice_J1'][x])**2 )
                     gain = (mise * cote)-mise
              elif (df_out['resultat'][x] == True) & (df_out['Winner'][x]==2):
                     cote =df_out['Cote J2'][x]
                     mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
                     gain = (mise * cote)-mise
              elif (df_out['resultat'][x] == False) & (df_out['Winner'][x]==1):
                     mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
                     gain = -mise
              else:    
                     mise = (mise_base)*((df_out['Indice_J1'][x])**2 )
                     gain = -mise  
              return gain
     
       df_out['Gain_pond'] = pd.Series(df_out.index).apply(f_gain_pond)


       def f_mise_pond(x):
              mise_base =1
   
              if (df_out['resultat'][x] == True) & (df_out['Winner'][x]==1):
                     mise = (mise_base)*((df_out['Indice_J1'][x])**2 )
              elif (df_out['resultat'][x] == True) & (df_out['Winner'][x]==2):
                     mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
              elif (df_out['resultat'][x] == False) & (df_out['Winner'][x]==1):
                     mise = (mise_base)*((df_out['Indice_J2'][x])**2 )
              else:
                     mise = (mise_base)*((df_out['Indice_J1'][x])**2 )  
              return mise
       df_out['Mise_pond'] = pd.Series(df_out.index).apply(f_mise_pond)



       df_out.dropna(axis=0, how="any", inplace=True)
       df_out =df_out.reset_index(drop=True)

       df_out.Joueur1=le.inverse_transform(df_out.Joueur1)
       df_out.Joueur2=le.inverse_transform(df_out.Joueur2)


       data_ROI = pd.DataFrame([])

       for i in np.arange(0.8, 1.4, 0.01):
              df_prob = df_out[(((df_out["Indice_J1"] > i) & (df_out["y_pred"] == 1)) | ((df_out["Indice_J2"] > i) & (df_out["y_pred"] == 2))) ]
              gain = df_prob['Gain'].sum()
              nb_paris=df_prob.shape[0]
              ROI = (gain/nb_paris)*100
        
              mise_pond = df_prob['Mise_pond'].sum()
              gain_pond = df_prob['Gain_pond'].sum()
              ROI_pond = (gain_pond/mise_pond)*100
        
              data_ROI = data_ROI.append(pd.DataFrame({'Threshold': i, 'nb_paris': nb_paris, 'Gain' : gain, 'ROI': ROI,'gain_pond':gain_pond,'ROI_pond':ROI_pond }, index=[0]), ignore_index=True)
    




       Joueur1 = st.selectbox('Sélectionner joueur n°1:',liste_joueur)
       st.write('Joueur n°1:', Joueur1)

       Cote_J1 = st.number_input('Sélectionner la côte du joueur n°1:')
       st.write('Côte du joueur n°1:', Cote_J1)

       Joueur2 = st.selectbox('Sélectionner joueur n°2:',liste_joueur)
       st.write('Joueur n°2:', Joueur2)

       Cote_J2 = st.number_input('Sélectionner la côte du joueur n°2:')
       st.write('Côte du joueur n°2:', Cote_J2)

       Surface =df.Surface.unique()
     
       liste_Surface = list(Surface)

       Surface = st.selectbox('Sélectionner La surface sur laquelle se joue le match :',liste_Surface)
       st.write('Match:', Surface)

       df.Joueur1=le.inverse_transform(df.Joueur1)
       df.Joueur2=le.inverse_transform(df.Joueur2) 

       dataJ1 = df[(df['Surface'] == Surface) & ((df['Joueur1'] == Joueur1) | (df['Joueur2'] ==Joueur1))].tail(1)
       dataJ2 = df[(df['Surface'] == Surface) & ((df['Joueur1'] == Joueur2) | (df['Joueur2'] ==Joueur2))].tail(1)

       if dataJ1['Joueur1'].values[0] == Joueur1:
              Elo_J1 = dataJ1["Elo J1"].values[0]
              Rank_J1 = dataJ1["Rank J1"].values[0]
              Surface_J1 = dataJ1["Surface_J1"].values[0]
              ratio_1m_J1 = dataJ1["ratio_1m_J1"].values[0]
              nb_1m_J1 = dataJ1["nb_1m_J1"].values[0]
              ratio_3m_J1 = dataJ1["ratio_3m_J1"].values[0]
              nb_3m_J1 = dataJ1["nb_3m_J1"].values[0]
       else:
              Elo_J1 = dataJ1["Elo J2"].values[0]
              Rank_J1 = dataJ1["Rank J2"].values[0]
              Surface_J1 = dataJ1["Surface_J2"].values[0]
              ratio_1m_J1 = dataJ1["ratio_1m_J2"].values[0]
              nb_1m_J1 = dataJ1["nb_1m_J2"].values[0]
              ratio_3m_J1 = dataJ1["ratio_3m_J2"].values[0]
              nb_3m_J1 = dataJ1["nb_3m_J2"].values[0]

       if dataJ2['Joueur1'].values[0] == Joueur2:
              Elo_J2 = dataJ2["Elo J1"].values[0]
              Rank_J2 = dataJ2["Rank J1"].values[0]
              Surface_J2 = dataJ2["Surface_J1"].values[0]
              ratio_1m_J2 = dataJ2["ratio_1m_J1"].values[0]
              nb_1m_J2 = dataJ2["nb_1m_J1"].values[0]
              ratio_3m_J2 = dataJ2["ratio_3m_J1"].values[0]
              nb_3m_J2 = dataJ2["nb_3m_J1"].values[0]
       else:
              Elo_J2 = dataJ2["Elo J2"].values[0]
              Rank_J2 = dataJ2["Rank J2"].values[0]
              Surface_J2 = dataJ2["Surface_J2"].values[0]
              ratio_1m_J2 = dataJ2["ratio_1m_J2"].values[0]
              nb_1m_J2 = dataJ2["nb_1m_J2"].values[0]
              ratio_3m_J2 = dataJ2["ratio_3m_J2"].values[0]
              nb_3m_J2 = dataJ2["nb_3m_J2"].values[0]
    




       Proba_elo= 1 / (1 + 10 ** ((Elo_J1 - Elo_J2) / 400))

       if Surface == "Clay":
              Surface= 0
       elif Surface == "Hard":
              Surface = 2
       else:
              Surface = 1

       Joueur1=le.transform([Joueur1])[0]
       Joueur2=le.transform([Joueur2])[0]

     
       data = [[Surface, Proba_elo, Joueur1, Joueur2, Cote_J1, Cote_J2, Rank_J1, Rank_J2, Elo_J1,Elo_J2, Surface_J1, Surface_J2, ratio_1m_J1, nb_1m_J1, ratio_1m_J2, nb_1m_J2, ratio_3m_J1, nb_3m_J1,ratio_3m_J2, nb_3m_J2]]

     
     
     
       column_names = ["Surface", "proba_elo","Joueur1","Joueur2", "Cote J1","Cote J2", "Rank J1", "Rank J2","Elo J1", "Elo J2", "Surface_J1", "Surface_J2", "ratio_1m_J1", "nb_1m_J1", "ratio_1m_J2", "nb_1m_J2", "ratio_3m_J1", "nb_3m_J1", "ratio_3m_J2"," nb_3m_J2" ]
       data = pd.DataFrame(data, columns=column_names)
       data  = data.to_numpy()
       data =scaler.transform(data) 

       model_pred = keras.models.load_model("my_model6.h5")



       if st.button("Prediction"):

              pred1 = model_pred.predict(data)
              pred2 = np.array([1-i for i in pred1])
              y_proba=np.c_[pred1,pred2]
     
              Joueur1=le.inverse_transform([Joueur1])[0]
              Joueur2=le.inverse_transform([Joueur2])[0]

     

              df_proba = pd.DataFrame ({'Joueur':  [Joueur1, Joueur2],'Probabilité': [round(pred1[0][0],2),round(pred2[0][0],2)]})
              df_indice = pd.DataFrame ({'Joueur':  [Joueur1, Joueur2],'Indice': [round((pred1[0][0])*(Cote_J1),2),round(pred2[0][0]*Cote_J2,2)]})

              fig, ax = plt.subplots(figsize=(9, 4))
              sns.barplot(x="Probabilité", y="Joueur", data=df_proba)
              ax.bar_label(ax.containers[0])
              st.pyplot(fig)

              fig, ax = plt.subplots(figsize=(9, 4))
              sns.barplot(x="Indice", y="Joueur", data=df_indice)
              ax.bar_label(ax.containers[0])
              st.pyplot(fig) 

     
             
elif genre == 'Bilan':
     st.write('''
            ### Bilan
            Ce projet fût un super exercice permettant de compléter notre formation. En effet, il nous a permis d’une part d’approfondir nos connaissance et d'une autre part nous a appris à trouver des solutions au delà de ce qui a été vu durant formation. 
            
            Nous avons pu revoir les différents modèles de classification ainsi que leurs méthodes de tuning des paramètres. 
            
            Lors de ce projet nous avons pu mesurer l’importance de la features engineering dans l'obtention de meilleurs résultats. Et d’une manière plus générale nous avons pu comprendre que dans tout projet de data la préparation, le nettoyage et la mise en forme des données est primordiale. Cette partie est au moins aussi importante que le choix du modèle. 
            
            Pour être performant sur le prétraitement de notre data nous avons pu réaliser aussi que la connaissance métier est indispensable dans la réussite d’un projet data. 
            
            ''')

elif genre == 'Pour Aller Plus Loin':
     st.write('''
            ### Pour Aller Plus Loin
            Pour aller plus loin dans ce projet on pourrait utiliser des nouvelles features comme l'historique de victoire entre 2 joueurs.

            Une approche complémentaire peut être de combiner le modèle avec du sentiment analysis afin d’ajouter des avis d’expert sur des pronostics.

            On pourrait également procéder à du web scraping afin d'améliorer l’automatisation d’ajout de nouvelles données.

            Une autre possibilité serait d'exploiter le nombre de jeux et de sets pour notamment utiliser les paris avec handicap pour obtenir de meilleures côtes. Ces types de paris sont généralement plus plébiscités chez les parieurs pro. 
            
            ''')

st.sidebar.subheader("Auteurs :")
st.sidebar.text("Valentin GOUZY")
st.sidebar.text("Sofiane OUCHENE")


        
