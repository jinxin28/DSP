import streamlit as st

def display_overview():
    # Heading
    st.write("### Welcome to the Crop Price Dashboard for Malaysian Farmers")
    
    # Introduction to crop prices and their importance
    st.write(
        """
        As a farmer, understanding crop prices is essential for planning the best times to plant, harvest, and sell your crops. Prices can change due to several factors such as weather patterns, market demand, and seasonal trends.
        
        In this dashboard, we use **Time Series Analysis** to predict future crop prices. Time series analysis looks at historical data to identify trends and patterns, which can help forecast future prices. By understanding these trends, you can make informed decisions about when to plant or sell your crops to maximize your profits.
        """
    )

    # Image of time series or crop price trends (optional)
    st.image("https://www.world-grain.com/ext/resources/2023/12/05/wheat-markets-prices_cr-BILLIONPHOTOS.COM--STOCK.ADOBE.COM_e.jpg?height=635&t=1703252509&width=1200", width=400)  # Replace with an actual URL

    # Step-by-step guide to using the platform
    st.write("### How to Use This Dashboard: A Simple Guide for Farmers")
    
    st.write(
        """
        1. **Start by Visiting the Template Page:** Before uploading your data, visit the **Template** page from the sidebar. This page provides a sample template and explains the required format for your CSV file.
        
        2. **Upload Your CSV File:** After understanding the required format, upload your crop price data in CSV format by clicking the 'Proceed to upload CSV' button in the sidebar.
        
        3. **Clean and Explore Your Data:** Once your data is uploaded, click the 'Data Cleaning & EDA' button. Here, we clean and prepare the data by removing any errors or missing information that could affect the time series predictions.
        
        4. **Predict Future Prices with Time Series:** After the data is cleaned, click the 'Price Prediction' section. Using time series analysis, we will predict future crop prices based on historical data trends. This will help you plan for the next season's crops and sales.
        
        5. **Analyze Results:** Once the predictions are ready, you can view graphs and forecasts that will guide your decisions in planting and selling crops.
        """
    )

    # Quick tips or advice for farmers
    st.write("### Quick Tips to Get the Most Out of This Tool")
    st.write(
        """
        - **Ensure Accurate and Timely Data:** The more recent and accurate your historical crop price data, the better the predictions.
        - **Look for Seasonal Trends:** Time series analysis can identify seasonal patterns, so you can plan for crops that are in high demand during specific times of the year.
        - **Use Predictions to Plan Ahead:** By predicting future crop prices, you can plan your planting and harvesting schedule to take advantage of the best market conditions.

        Time series analysis is a powerful tool that helps farmers like you make better decisions. With the help of this tool, you can optimize your farming operations and improve your income.
        """
    )
