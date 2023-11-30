import numpy as np
import streamlit as st
import pickle

# Load the trained model
loaded_model = pickle.load(open('C:/Users/suyog/OneDrive/Desktop/diabetes/trained_model_.sav', 'rb'))

# Load the scaler
scaler = pickle.load(open('C:/Users/suyog/OneDrive/Desktop/diabetes/standard.pkl', 'rb'))

# creating function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshape)

    # make predictions using the loaded model
    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        return "The patient doesn't have diabetes"
    else:
        return "The patient has Diabetes"

def main():
    # giving a title
    st.title('Diabetes prediction web App')

    # getting the input data from the user
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('BloodPressure value')
    SkinThickness = st.text_input('SkinThickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetsPedigreeFuction = st.text_input('DiabetesPedigreeFunction level')
    Age = st.text_input('Age of person')

    # code for prediction
    diagnosis = ""

    # creating a button for prediction
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetsPedigreeFuction, Age])
    st.success(diagnosis)

if __name__ == '__main__':
    main()
