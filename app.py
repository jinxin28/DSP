import streamlit as st
import pandas as pd
from data_cleaning import data_cleaning_and_eda
from data_modelling import price_prediction
import overview  # Import the overview.py module
import template  # Import the template.py module

# Set page configuration for the Streamlit app
st.set_page_config(
    page_title="Crop Data Analysis Dashboard",  # Title for the browser tab
    page_icon="🌱",  # An emoji icon to display in the browser tab
    layout="wide",  # Layout choice: centered or wide
    initial_sidebar_state="expanded"  # Sidebar state when the app is first loaded
)

# Now you can proceed with the rest of your Streamlit code
st.title("Crop Data Analysis Dashboard")
st.write("Welcome to the Crop Data Dashboard. Here you can explore and predict crop prices in Malaysia.")

# Initialize session state for navigation and file upload
if "page" not in st.session_state:
    st.session_state.page = "Overview"  # Default to Overview page when app opens

# Display images immediately upon loading the app (under the title)
overview_image = "https://images.unsplash.com/photo-1560493676-04071c5f467b?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Replace with your image path or URL
data_cleaning_image = "https://cdn.prod.website-files.com/5a00e7aa079aa40001b3c4fb/5d5c22e040c6beab16860e8e_data-cleaning-thumb.png"  # Replace with your image path or URL
price_prediction_image = "https://storage.googleapis.com/kaggle-datasets-images/4902880/8260604/25024b490d7a911746a1c670c948399d/dataset-cover.jpg?t=2024-04-30-17-07-14"  # Replace with your image path or URL

# Display the appropriate image based on the current page
if st.session_state.page == "Overview":
    st.image(overview_image, caption="Overview Page Image", use_column_width=True)
elif st.session_state.page == "Data Cleaning & EDA":
    st.image(data_cleaning_image, caption="Data Cleaning & EDA Page Image", use_column_width=True)
elif st.session_state.page == "Price Prediction":
    st.image(price_prediction_image, caption="Price Prediction Page Image", use_column_width=True)

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")

    # Overview button
    if st.button("Overview"):
        st.session_state.page = "Overview"

    # Template button
    if st.button("Template"):
        st.session_state.page = "Template"

    # Upload CSV file button - Always Displayed
    st.write("### Upload Your CSV File")
    st.session_state.uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Data Cleaning & EDA button
    if st.button("Data Cleaning & EDA"):
        st.session_state.page = "Data Cleaning & EDA"

    # Price Prediction button
    if st.button("Price Prediction"):
        st.session_state.page = "Price Prediction"

# Page Content
if st.session_state.page == "Overview":
    st.write("### Overview")
    overview.display_overview()

elif st.session_state.page == "Template":
    st.write("### Template")
    template.display_template()

elif st.session_state.page == "Data Cleaning & EDA":
    if st.session_state.uploaded_file is not None:
        df = pd.read_csv(st.session_state.uploaded_file)
        
        # Display original data preview
        st.write("### Data Preview")
        st.write(df.head())

        # Perform data cleaning and EDA
        df_cleaned = data_cleaning_and_eda(df)

        if isinstance(df_cleaned, pd.DataFrame):
            st.session_state.df_cleaned = df_cleaned  # Save cleaned data in session state
            st.write("### Cleaned Data")
            st.write(df_cleaned.head())
        else:
            st.error("Data cleaning function did not return a valid DataFrame.")
    else:
        st.warning("Please upload a CSV file to proceed.")
    
elif st.session_state.page == "Price Prediction":
    if "df_cleaned" in st.session_state and st.session_state.df_cleaned is not None:
        st.write("### Cleaned Data for Price Prediction")
        st.write(st.session_state.df_cleaned.head())  # Display cleaned data

        # Perform price prediction
        price_prediction(st.session_state.df_cleaned)
    else:
        st.warning("Please complete the Data Cleaning & EDA step first.")
