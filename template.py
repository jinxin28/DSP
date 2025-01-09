import streamlit as st
import pandas as pd
import requests
import io  # To handle bytes data as file-like objects

def display_template():
    """
    Display the Template page with options to download a predefined sample CSV, view data, and explanation.
    """

    # URL of the hosted template file
    template_url = "https://raw.githubusercontent.com/jinxin28/DSP/master/3_crops.csv"  # Use raw GitHub link

    try:
        # Download the file from the URL
        response = requests.get(template_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        template_data = response.content  # Get the content as bytes

        # Provide download functionality
        st.download_button(
            label="Download Template CSV",
            data=template_data,
            file_name="template.csv",
            mime="text/csv"
        )

        # Convert the bytes data to a file-like object and read into a DataFrame
        df_template = pd.read_csv(io.BytesIO(template_data))
        st.write("#### Template Data Preview")
        st.write(df_template.head())

        # Explanation of rows and columns
        st.write("#### Explanation of the Template")
        st.markdown(
            """
            - **Item**: The crop/item (e.g., Papayas).
            - **Year**: The year the data corresponds to.
            - **Months**: The month the data corresponds to.
            - **Value**: The value recorded for that month in LCU/tonne.
            """
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the template file: {e}")
    except pd.errors.ParserError:
        st.error("Error parsing the template file. Ensure it is in valid CSV format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

