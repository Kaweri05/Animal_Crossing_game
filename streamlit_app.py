
import streamlit as st
import pandas as pd
import pickle

# Load the model, preprocessor, and feature names
with open('random_forest_model.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('feature_names_out.pkl', 'rb') as file:
    feature_names_out = pickle.load(file)

st.title('Animal Crossing Item Sell Price Predictor')
st.write('Predict the sell price category (Low, Medium, High) for your items!')

# Input fields for new data
buy_price = st.number_input('Enter Buy Price (e.g., 490)', min_value=0, value=100)
item_type = st.selectbox('Select Item Type',
                           ['AccessoryEye', 'AccessoryMouth', 'AccessoryEyeMouthInvisibleNose', 
                            'AcceEyeMouth', 'AccessoryMouthEarJaw', 'AccessoryOneEye'])

if st.button('Predict Sell Category'):
    # Create new data DataFrame
    new_data_input = pd.DataFrame({
        'Buy': [buy_price],
        'Type': [item_type]
    })

    # Apply preprocessing
    X_new_transformed = preprocessor.transform(new_data_input)
    X_new_processed_df = pd.DataFrame(X_new_transformed.toarray(), columns=feature_names_out)

    # Make prediction
    prediction = rf_classifier.predict(X_new_processed_df)
    st.success(f'The predicted Sell Price Category is: **{prediction[0]}**')

st.write('---')
st.write('**Note:** This is a simplified predictor based on Buy Price and Item Type.')
import pickle

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)
