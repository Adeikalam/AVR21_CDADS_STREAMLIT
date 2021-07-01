import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt



df = pd.read_csv("train.csv", index_col = 0)

page = st.sidebar.radio("", options = ['Présentation', 'Modélisation', 'Conclusion'])

if page == 'Présentation':
    
    st.title("Mon Projet de ML")
    
    image = plt.imread("titanic.jpg")
    
    st.image(image)

    st.markdown("""
                Ce projet de Machine Learning a été fait sur le dataset du 
                [titanic](https://www.kaggle.com/c/titanic/overview).
                
                Nous allons tester 3 modèles:
                * Regression Logistique
                * KNN
                * Decision Tree
                
                """)
                
    
    
    
if page == 'Modélisation':
    
    # Nettoyage du dataset
    
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
    
    # Gestion des NAs
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Encodage des variables catégorielles
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
    
    ohe = pd.get_dummies(df['Embarked'])
    df = pd.concat([df, ohe], axis = 1)
    df = df.drop('Embarked', axis = 1)
    
    # Nettoyage fini
    
    st.write(df)
    
    X = df.drop('Survived', axis = 1)
    y = df['Survived']
    
    
    # Modelisation
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    
    st.header("Veuillez choisir un modèle:")
    
    choix_modele = st.radio("", options = ['Régression Logistique',
                                            'KNN',
                                            'Decision Tree']) 
    
    if choix_modele == 'Régression Logistique':
        model = LogisticRegression()
    
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        
        st.write(score)
        
    if choix_modele == 'KNN':
        model = KNeighborsClassifier()
    
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        
        st.write(score)
    
    if choix_modele == 'Decision Tree':
        model = KNeighborsClassifier()
    
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        
        st.write(score)
    
    def test_model(choix_modele):
        if choix_modele == 'Régression Logistique':
            model = LogisticRegression()
            
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            
            st.write(score)
            
        if choix_modele == 'KNN':
            model = KNeighborsClassifier()
            
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            
            st.write(score)
            
        if choix_modele == 'Decision Tree':
            model = DecisionTreeClassifier()
            
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            
            st.write(score)

    test_model(choix_modele)


















