import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
from langchain import PromptTemplate, LLMChain
import os
from huggingface_hub import InferenceApi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Define the pages
def main_page():
    st.header("Healthcare")
    st.info("Please configure the settings on the 'Configuration' page.")



# Load the saved model and scaler
loaded_model = pickle.load(open('C:/Users/suyog/OneDrive/Desktop/final/trained_model_.sav', 'rb'))
scaler = pickle.load(open('C:/Users/suyog/OneDrive/Desktop/final/standard.pkl', 'rb'))


# Create function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshape)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "The patient doesn't have diabetes"
    else:
        return "The patient has diabetes"


def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('BloodPressure value')
    SkinThickness = st.text_input('SkinThickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function level')
    Age = st.text_input('Age of person')

    # Code for prediction
    diagnosis = ""

    # Creating a button for prediction
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction(
            [Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    st.success(diagnosis)


def bmi_calculator():
    st.title("Body Mass Index (BMI) Calculator")

    # Getting the input data from the user
    weight = st.text_input("Enter your weight in kg:")
    height = st.text_input("Enter your height in meter:")

    # Check if inputs are provided
    if weight and height:
        try:
            # Convert inputs to float for calculation
            weight = float(weight)
            height = float(height)

            # BMI formula: BMI = (weight in pounds * 703) / (height in inches)^2
            #bmi = (weight * 703) / (height * height)
            bmi = weight / (height * height)

            # Display the calculated BMI
            if st.button(""):
                print(bmi)
            st.success(f"Your BMI is: {bmi:.2f}")

            # Optional: Give a category based on BMI value
            if bmi < 18.5:
                st.info("You are underweight.")
            elif 18.5 <= bmi < 24.9:
                st.info("You have a normal weight.")
            elif 25 <= bmi < 29.9:
                st.info("You are overweight.")
            else:
                st.info("You are obese.")
        except ValueError:
            st.error("Please enter valid numeric values for weight and height.")
    else:
        st.warning("Please enter both weight and height.")

'''------------------------------------------------------------------------------------------------------'''

def ChatBot():
# Set the Hugging Face API token
    api_token = "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"
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = api_token


# Streamlit App title
    st.title("Healthcare Query Bot")

# User input for the question
    question = st.text_input("Enter your Health Issues:", )

# Hugging Face Inference Setup
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    api = InferenceApi(repo_id=repo_id, token=api_token)

# Define Langchain template
    template = """Question: {question}
Answer: Let's think step by step.
"""
    llm_prompt = PromptTemplate(template=template, input_variables=["question"])

# Streamlit button to trigger the request
    if st.button("Submit Query"):
        try:
        # Call the model using the Langchain and Hugging Face integration
            llm_chain = LLMChain(llm=api, prompt=llm_prompt)
            response = llm_chain.invoke(question)

        # Display the response
            st.write(f"Answer: {response}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Sidebar for navigation
with st.sidebar:
    selected_page = option_menu(
        "Menu 📃",
        ("Diabetic Prediction", "BMI", "ChatBot"),
        icons=['house', 'gear', '🤖']
    )

# Load the appropriate page based on sidebar selection
if selected_page == "Diabetic Prediction":
    main()

elif selected_page == "BMI":
    bmi_calculator()

elif selected_page == "ChatBot":
    ChatBot()
