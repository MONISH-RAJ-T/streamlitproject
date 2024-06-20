import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
st.title("Simple Streamlit ML Model")

 
with st.sidebar:
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:

            return json.load(f)
    lottie_codeing = load_lottiefile("C:\\Users\\User\\Downloads\\Animation - 1712585240627.json")

    st.title(" list of contents")
    st_lottie(lottie_codeing,speed=1,reverse=False,loop=True,quality="low",height=None,width=None,key=None,)
    choice=st.radio("Menu", ["Home", "Train Model", "Predict"])

if choice == "Train Model":
    st.header("Train a Machine Learning Model")

    # Upload CSV file
    uploaded_file = st.file_uploader("/content/drive/MyDrive/Social_Network_Ads.csv", type=["csv"])

    if uploaded_file is not None:
        st.write("File uploaded successfully.")

        # Read the data
        df = pd.read_csv("C:\\Users\\User\\Downloads\\Social_Network_Ads.csv")

        # Display the first few rows of the dataframe
        st.write("Sample data:")
        st.write(df.head())

        label_encoder = LabelEncoder()
        for column in df.columns:
            if df['Gender'].dtype == 'object':
                df['Gender'] = label_encoder.fit_transform(df['Gender'])
        
        # Select target column
        target_column = st.selectbox("Purchased", df.columns)

        # Display heatmap for the correlation matrix
        st.subheader("Correlation Heatmap")
        corr_matrix = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Split data into features and target
        X = df.drop(columns=["Purchased"])
        y = df["Purchased"]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Model Accuracy:", accuracy)

        # Feature importance
        st.header("Feature Importance")
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importance)

if choice == "Predict":
    with open("C:\\Users\\User\\Downloads\\trained_model2.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
    def predict_emission(UserID, Gender, Age, EstimatedSalary):
        features = np.array([UserID, Gender, Age, EstimatedSalary])
        features = features.reshape(1,-1)
        purchased = model.predict(features)
        return purchased[0]
        
    
 
       
    UserID= st.number_input('User_ID')
    Gender= st.number_input('Gender')
    Age= st.number_input('Age')
    EstimatedSalary = st.number_input('EstimatedSalary')    

        # Prediction button
if st.button('Predict'):
    # Predict EMISSION
    purchased = predict_emission(UserID, Gender, Age, EstimatedSalary)

    st.write(f"Predicted purchased: {purchased}")


        



        




