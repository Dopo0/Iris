import streamlit as st
import pickle
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,f1_score

def train():
    
    columns_to_scale = ['sepal length (cm)', 'sepal width (cm)','petal length (cm)', 'petal width (cm)']

    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    pickle.dump(scaler, open('./pages/obj/mlp_scaler.pkl', 'wb'))
    
    X = df.drop(columns=['Species', 'Species_Name'])
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = MLPClassifier(random_state=42,
                      max_iter=max_iter,
                      hidden_layer_sizes=(10, 10,),
                      n_iter_no_change=100,
                      early_stopping=early_stopping,
                      #verbose=True
                         ).fit(X_train, y_train)
    pickle.dump(model, open('./pages/obj/mlp_model.pkl', 'wb'))
    
    st.subheader('Evaluation')
    tab1, tab2, tab3 = st.tabs(["loss_curve ", "validation_scores",'cm'])
    with tab1:
        loss_curve = model.loss_curve_
        st.pyplot(loss_curve)
            
            
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()


def main():
    st.set_page_config(page_title="Iris dataset analysis",page_icon="🍃",)
    st.title("Iris MLP model")
    
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["Species"] = data.target
    df["Species_Name"] = df["Species"]
    for index, row in df.iterrows():
        species = int(row["Species"])
        df.loc[index, "Species_Name"] = data.target_names[species]
    df = df.astype({"Species": "category"})
    st.subheader('Dataset samples')
    st.write(df.sample(15))
    st.subheader('Visualization')
    tab1, tab2, tab3 = st.tabs(["3D Scatter", "Boxplot",'Violin'])
    with tab1:
        fig = px.scatter_3d(df,
                        x="petal length (cm)",
                        y="petal width (cm)",
                        z="sepal length (cm)",
                        color="Species_Name",
                        height=500)
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with tab2:
        fig = px.box(df, x=['sepal length (cm)', 'sepal width (cm)',
                        'petal length (cm)', 'petal width (cm)'],
                 color="Species_Name")
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with tab3:
        fig = px.violin(df, y=['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)'],
                color="Species_Name",
                points='all', box=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)
    
    max_iter=st.sidebar.number_input('max_iter',min_value=100,step=50)
    options = ((10,10,),(20,20,),(20,10,),(30,10,))
    hidden_layer_sizes= st.sidebar.selectbox('hidden layer',options = options)
    n_iter_no_change = st.sidebar.number_input('n_iter_no_change',min_value=100,step=10)
                    
                      #early_stopping=True
    early_stopping = st.sidebar.checkbox('Enable early stopping',value=True)
    verbose = st.sidebar.checkbox('Enable verbose',value=False)
    btn_retrain = st.sidebar.button("Retrain")
    
    
    if btn_retrain:
        columns_to_scale = ['sepal length (cm)', 'sepal width (cm)','petal length (cm)', 'petal width (cm)']

        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        pickle.dump(scaler, open('./pages/obj/mlp_scaler.pkl', 'wb'))

        X = df.drop(columns=['Species', 'Species_Name'])
        y = df["Species"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        model = MLPClassifier(random_state=42,
                          max_iter=max_iter,
                          hidden_layer_sizes=hidden_layer_sizes,
                          n_iter_no_change=n_iter_no_change,
                          early_stopping=early_stopping,
                          verbose=verbose
                             ).fit(X_train, y_train)
        pickle.dump(model, open('./pages/obj/mlp_model.pkl', 'wb'))

        st.subheader('Evaluation')
        tab1, tab2, tab3 = st.tabs(["loss_curve ", "validation_scores",'Confusion Matrix'])
        with tab1:
            loss_curve = model.loss_curve_
            fig, ax = plt.subplots()
            ax.plot(loss_curve)
            st.plotly_chart(fig)
            
        with tab2:
            validation_scores = model.validation_scores_
            fig, ax = plt.subplots()
            ax.plot(validation_scores)
            st.plotly_chart(fig)
        
        with tab3:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            #st.plotly_chart(fig, theme=None, use_container_width=True)
            st.pyplot(fig)
        
        st.sidebar.subheader('Scores')
        col1, col2, col3 = st.columns(3)
        
        y_pred = model.predict(X_test)
        st.sidebar.success(f'Train score achived {model.score(X_train, y_train)}')
        st.sidebar.success(f'Test score achived {model.score(X_test, y_test)}')
        #st.sidebar.success(f'F1-score achived {f1_score(y_test, y_pred,average=None)}')
    
        
        
           


if __name__=="__main__":
    main()