# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:21:01 2024

@author: ayank
"""
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/ayank/MDP/MDP/diabetes_model.sav', 'rb'))

# creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # giving a title for the webpage
    st.title('Diabetes Prediction Web App')

    # giving the user interface (input data from the user)
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure level')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for the prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to appropriate data types
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = 'Please enter valid input values'
        
        st.success(diagnosis)

if __name__ == '__main__':
    main()

        
        
        
        
        
        

    
    