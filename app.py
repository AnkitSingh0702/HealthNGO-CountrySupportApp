import streamlit as st
import numpy as np
import joblib

# Load models
with open('sc.joblib', 'rb') as file:
    scale = joblib.load(file)
with open('pca1.joblib', 'rb') as file:
    pca = joblib.load(file)
with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)

# Prediction function
def prediction(input_list):
    scaled_input = scale.transform([input_list])
    pca_input = pca.transform(scaled_input)
    output = model.predict(pca_input)[0]

    if output == 0:
        return "Developing"
    elif output == 1:
        return "Developed"
    else:
        return "Underdeveloped"

# Main App
def main():
    st.set_page_config(page_title="Help NGO Foundation", layout="centered")

    st.title("🌍 Help NGO Foundation")
    st.markdown("This application predicts the **status of a country** based on socio-economic and health factors.")

    # Group inputs into sections
    with st.expander("📊 Economic Indicators", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gdp = st.number_input("GDP per population", min_value=0.0, value=15000.0)
            inc = st.number_input("Per capita income", min_value=0.0, value=12000.0)
            imp = st.number_input("Imports (% of GDP)", min_value=0.0, value=25.0)
            exp = st.number_input("Exports (% of GDP)", min_value=0.0, value=30.0)
        with col2:
            inf = st.number_input("Inflation rate (%)", min_value=0.0, value=4.0)

    with st.expander("❤️ Health & Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            hel = st.number_input("Health expenditure (% of GDP)", min_value=0.0, value=6.0)
            ch_m = st.number_input("Child mortality (per 1000 births < 5 yrs)", min_value=0.0, value=20.0)
        with col2:
            fer = st.number_input("Fertility rate (children per woman)", min_value=0.0, value=2.1)
            lf = st.number_input("Life expectancy (years)", min_value=0.0, value=72.0)

    in_data = [ch_m, exp, hel, imp, inc, inf, lf, fer, lf]

    if st.button("🔮 Predict"):
        response = prediction(in_data)
        if response == "Developed":
            st.success(f"✅ Prediction: {response}")
        elif response == "Developing":
            st.warning(f"⚠️ Prediction: {response}")
        else:
            st.error(f"❌ Prediction: {response}")

if __name__ == "__main__":
    main()
