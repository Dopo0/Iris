import streamlit as st
import pickle
import pandas as pd
import os

import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    st.title("Iris classification")
    sepal_len = st.sidebar.slider("sepal length",4.3,7.9,5.8)
    sepal_wid = st.sidebar.slider('sepal width',2.0,4.4,3.0)
    petal_len = st.sidebar.slider('petal length',1.0,6.9,4.3)
    petal_wid = st.sidebar.slider('petal width',0.1,2.5,1.3)
    
    column_names=["sepal length (cm)",'sepal width (cm)','petal length (cm)','petal width (cm)']
    x =  pd.DataFrame([[sepal_len,sepal_wid,petal_len,petal_wid]],columns=column_names)
    

    model = pickle.load(open('pages/obj/mlp_model.pkl', 'rb'))
    scaler = pickle.load(open('pages/obj/mlp_scaler.pkl', 'rb'))
    btn_submit = st.sidebar.button("Classify")

    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["Species"] = data.target
    df["Species_Name"] = df["Species"]
    for index, row in df.iterrows():
        species = int(row["Species"])
        df.loc[index, "Species_Name"] = data.target_names[species]
    df = df.astype({"Species": "category"})

    
    fig = px.scatter_3d(df,x="petal length (cm)",y="petal width (cm)",z="sepal length (cm)",color="Species_Name",height=500)

   
    fig.add_trace(px.scatter_3d(pd.DataFrame([[petal_len, petal_wid, sepal_len]], columns=["x", "y", "z"]),
                                x="x", y="y", z="z").data[0])
    fig.data[-1].update(marker=dict(size=12, color='yellow'))
    st.plotly_chart(fig, theme=None, use_container_width=True)
   
    if btn_submit:
        x[column_names] = scaler.transform(x[column_names])
        result = model.predict(x)
        names = {0:'setosa',1:'versicolor',2:'virginica'}
        st.sidebar.subheader("Result")
        y = names[result[0]]
        st.sidebar.success(f'Species: {y}')

if __name__=="__main__":
    main()