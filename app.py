import streamlit as st
import numpy as np 
import pandas as pd 
import joblib 



# first lets load the instances that were created 
with open('sc.joblib' , 'rb') as file:
    scale = joblib.load(file)
with open('pca1.joblib' , 'rb') as file:
    pca = joblib.load(file)
with open('final_model.joblib' , 'rb') as file:
    model = joblib.load(file)

def prediction(input_list):
    scaled_input = scale.transform([input_list])
    pca_input = pca.transform(scaled_input)
    output = model.predict(pca_input)[0]

    if output == 0:
        return 'Developing'
    elif output == 1:
        return 'Developed'
    else:
        return'Underdeveloped'

def main():
    st.title('Help NGO Foundation')
    st.subheader('This application will give the status of a country based on socio-economic and health factor ')

    gdp = st.text_input("Enter the GDP per population of a country ")
    inc  = st.text_input("Enter the  per capita income  of a country ")
    imp = st.text_input("Enter the imports in terms of % of gdp")
    exp = st.text_input("Enter the export in terms of % of gdp")
    inf  = st.text_input("Enter the  inflation rate  of a country (%)")

    hel = st.text_input("Enter the expensiture on health in terms of % of gdp ")
    ch_m = st.text_input("Enter the number of deaths per 1000 births for less than  5 years")
    fer  = st.text_input("Enter the average children born to a women in a country ")
    lf = st.text_input("Enter the average life expenctancy of a country")
    in_data = [ch_m , exp , hel , imp , inc , inf , lf , fer , lf]
    if st.button('Predict'):
        response = prediction(in_data)
        st.success(response)

if __name__ =='__main__':
    main()
    
