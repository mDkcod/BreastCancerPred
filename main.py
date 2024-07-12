
#H:\ananconda1\envs\myenv2\lib\site-packages\google\protobuf\internal\__init__.py
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import cv2
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts
import plotly.graph_objects as go
from PIL import Image
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import json
from streamlit_folium import folium_static
import folium as fl
import pickle as pkl
import matplotlib.pyplot as plt
#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
scal=MinMaxScaler()
import warnings

model=pkl.load(open("final_model","rb"))

st.set_page_config(layout="centered", page_icon="⚕️",initial_sidebar_state="expanded",page_title="Disease Diagnosis")


dt= st.sidebar.write("WELCOME!!!!!Choose an action from below")

menu= ["Home","Breast Cancer", "Heart Disease"]
selection= st.sidebar.selectbox("Select Action--", menu)

select= st.sidebar.file_uploader("Upload New File", type=["csv", "xlsx"])


if selection== "Home":
    with st.container():

        st.title("Disease Diagnosis & Prediction")
        st.subheader('"Integrating AI in healthcare"')
        col1, col2, col3 = st.columns(3)
        col1.metric(" Model Accuracy", "70 °F", "1.2 °F")
        col2.metric("Chronic illnesses", "9 mph", "-8%")
        col3.metric("Humidity", "86%", "4%")



elif selection== "Breast Cancer":
    filepath = './saved_model'
    model1 = load_model(filepath, compile=True)

    with st.container():

        st.title("Breast Cancer Diagnosis")

        uploaded_file = st.file_uploader("Upload New File", type=["jpg", "png"])


        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            original_image = np.array(original_image)

            st.image(original_image, channels="BGR", caption='Mammographic image', use_column_width=True)

            input_tensor = tf.expand_dims(original_image, axis=0)

            pred = model1.predict(input_tensor)
            classes = np.argmax(pred, axis=1)
            # st.write(classes)
            if st.button("Diagnose"):
                if classes == 1:
                    st.error('Warning! Possible Breast Cancer!'
                             ' Perform further tests')

                else:
                    st.success('No Malignancy detected!')

        # Load the model
        #if uploaded_file is not None:
            # Convert the file to an opencv image.
            #file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            #opencv_image = cv2.imdecode(file_bytes, 1)

            # Now do something with the image! For example, let's display it:
            #st.image(opencv_image, channels="BGR", caption='Mammographic image', use_column_width=True)

            #opencv_image = cv2.imdecode(file_bytes, 1)
            #img_resized = cv2.resize(opencv_image, (50, 50), interpolation=cv2.INTER_LINEAR)
            #input_tensor = tf.expand_dims(img_resized, axis=0)

            #pred = model1.predict(input_tensor)
            #classes = np.argmax(pred, axis=1)
            #st.write(classes)


            #n_img = cv2.imread(opencv_image, cv2.IMREAD_COLOR)
        #if select is not None:
            #image = select.read()
            #img = st.image(image, caption='Mammographic image', use_column_width=True)

            # Convert the file to an opencv image.
            #file_bytes = np.asarray(bytearray(select.read()), dtype=np.uint8)
            #opencv_image = cv2.imdecode(file_bytes, 1)

            #n_img = cv2.imread(img, cv2.IMREAD_COLOR)
            #n_img_resized = cv2.resize(opencv_image, (50, 50), interpolation=cv2.INTER_LINEAR)

            # resize tensor to 224 x 224
            # tensor = tf.image.resize(n_img, [50, 50])
            #input_tensor = tf.expand_dims(n_img_resized, axis=0)

            #pred = model.predict(input_tensor)
            #classes = np.argmax(pred, axis=1)
            #print(classes)


elif selection== "Heart Disease":
    html_temp = """ 
        <div style ="background-color:pink;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Healthy Heart</h1> 
        </div> 
        """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)


    def preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):

        # Pre-processing user input
        if sex == "male":
            sex = 1
        else:
            sex = 0

        if cp == "Typical angina":
            cp = 0
        elif cp == "Atypical angina":
            cp = 1
        elif cp == "Non-anginal pain":
            cp = 2
        elif cp == "Asymptomatic":
            cp = 2

        if exang == "Yes":
            exang = 1
        elif exang == "No":
            exang = 0

        if fbs == "Yes":
            fbs = 1
        elif fbs == "No":
            fbs = 0

        if slope == "Upsloping: better heart rate with excercise(uncommon)":
            slope = 0
        elif slope == "Flatsloping: minimal change(typical healthy heart)":
            slope = 1
        elif slope == "Downsloping: signs of unhealthy heart":
            slope = 2

        if thal == "fixed defect: used to be defect but ok now":
            thal = 6
        elif thal == "reversable defect: no proper blood movement when excercising":
            thal = 7
        elif thal == "normal":
            thal = 2.31

        if restecg == "Nothing to note":
            restecg = 0
        elif restecg == "ST-T Wave abnormality":
            restecg = 1
        elif restecg == "Possible or definite left ventricular hypertrophy":
            restecg = 2

        user_input = [age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal]
        user_input = np.array(user_input)
        user_input = user_input.reshape(1, -1)
        user_input = scal.fit_transform(user_input)
        prediction = model.predict(user_input)

        return prediction

        # front end elements of the web page



    # following lines create boxes in which user can enter data required to make prediction
    age = st.selectbox("Age", range(1, 121, 1))
    sex = st.radio("Select Gender: ", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type', ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"))
    trestbps = st.selectbox('Resting Blood Sugar', range(1, 500, 1))
    restecg = st.selectbox('Resting Electrocardiographic Results', (
    "Nothing to note", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"))
    chol = st.selectbox('Serum Cholestoral in mg/dl', range(1, 1000, 1))
    fbs = st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes', 'No'])
    thalach = st.selectbox('Maximum Heart Rate Achieved', range(1, 300, 1))
    exang = st.selectbox('Exercise Induced Angina', ["Yes", "No"])
    oldpeak = st.number_input('Oldpeak')
    slope = st.selectbox('Heart Rate Slope', (
    "Upsloping: better heart rate with excercise(uncommon)", "Flatsloping: minimal change(typical healthy heart)",
    "Downsloping: signs of unhealthy heart"))
    ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', range(0, 5, 1))
    thal = st.selectbox('Thalium Stress Result', range(1, 8, 1))

    # user_input=preprocess(sex,cp,exang, fbs, slope, thal )
    pred = preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal)

    if st.button("Predict"):
        if pred[0] == 0:
            st.error('Warning! You have high risk of getting a heart attack!')

        else:
            st.success('You have lower risk of getting a heart disease!')

    st.sidebar.subheader("About App")

    st.sidebar.info("This web app is helps you to find out whether you are at a risk of developing a heart disease.")
    st.sidebar.info(
        "Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")
    st.sidebar.info("Don't forget to rate this app")

    feedback = st.sidebar.slider('How much would you rate this app?', min_value=0, max_value=5, step=1)

    if feedback:
        st.header("Thank you for rating the app!")
        st.info(
            "Caution: This is just a prediction and not doctoral advice. Kindly see a doctor if you feel the symptoms persist.")



