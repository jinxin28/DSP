import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Define the function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

# Function to create a Date column
def create_date(row):
    try:
        return datetime.strptime(f"{row['Year']}-{row['Months'][:3]}", '%Y-%b')
    except ValueError:
        return None

# File upload handling within Streamlit
def data_cleaning_and_eda(df):
    # Group data by Item (Papayas, Pineapples, Watermelons)
    crop_groups = df.groupby('Item')

    cleaned_crops = []

    # Visualize outliers and remove them for each crop
    for crop_name, crop_data in crop_groups:
        # Detect outliers using IQR method and add an 'Outlier' column to the crop_data
        crop_data['Outlier'] = detect_outliers_iqr(crop_data, 'Value')

        # Remove outliers from the dataset
        crop_data_cleaned = crop_data[~crop_data['Outlier']]

        # Append cleaned data to the list
        cleaned_crops.append(crop_data_cleaned)

    # Combine the cleaned crop datasets back into one DataFrame
    df_cleaned = pd.concat(cleaned_crops).sort_values(
        by=["Item", "Year", "Months"]
    ).reset_index(drop=True)

    # Drop 'Outlier' column
    df = df_cleaned.drop(columns=['Outlier'])

    # Define the list of months
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # Create a DataFrame with all years, months, and items
    unique_years = df["Year"].unique()
    unique_items = df["Item"].unique()
    full_year_month_item = pd.DataFrame(
        [(year, month, item) for year in unique_years for month in months for item in unique_items],
        columns=["Year", "Months", "Item"]
    )

    # Merge with the original data
    merged_df = pd.merge(
        full_year_month_item,
        df,
        how="left",
        on=["Year", "Months", "Item"]
    )

    # Interpolate missing 'Value' for each item (including the last year)
    merged_df["Value"] = (
        merged_df.groupby("Item")["Value"]
        .transform(lambda x: x.interpolate())
    )

    # Sort for clarity
    merged_df = merged_df.sort_values(by=["Item", "Year", "Months"]).reset_index(drop=True)

    # Apply the function to create the Date column
    merged_df['Date'] = merged_df.apply(create_date, axis=1)

    # Drop rows where the date conversion failed
    merged_df = merged_df.dropna(subset=['Date'])


    # Validate data types
    #st.write("### Data Types:")
    #st.write(merged_df.dtypes)

    # Create a function to process each crop's data
    def process_crop_data_daily(df_crop):
        # Ensure 'Date' is in datetime format
        df_crop['Date'] = pd.to_datetime(df_crop['Date'])

        # Create a daily date range for this crop
        daily_date_range = pd.date_range(
            start=df_crop['Date'].min(),
            end=df_crop['Date'].max() + pd.Timedelta(days=30),  # Extend by 30 days
            freq='D'
        )

        # Set 'Date' as the index for reindexing
        df_crop.set_index('Date', inplace=True)

        # Reindex to include all dates in the range
        df_crop_daily = df_crop.reindex(daily_date_range, fill_value=np.nan)

        # Perform linear interpolation for 'Value'
        df_crop_daily['Value'] = df_crop_daily['Value'].interpolate(method='linear')

        # Round 'Value' to 2 decimal places
        df_crop_daily['Value'] = df_crop_daily['Value'].round(2)

        # Reset the index to restore 'Date' as a column
        df_crop_daily.reset_index(inplace=True)
        df_crop_daily.rename(columns={'index': 'Date'}, inplace=True)

        return df_crop_daily

    # Create a full daily date range for each crop, adding 30 days beyond the max date
    full_date_range = pd.DataFrame({
        'Date': pd.date_range(start=merged_df['Date'].min(), 
                            end=merged_df['Date'].max() + pd.Timedelta(days=30))
    })

    expanded_data = []
    for crop in merged_df['Item'].unique():
        crop_data = merged_df[merged_df['Item'] == crop]
        crop_full_range = pd.merge(full_date_range, crop_data, how='left', on='Date')
        crop_full_range['Item'] = crop  # Reassign crop name in case of missing rows
        expanded_data.append(crop_full_range)

    # Combine all expanded datasets
    df_expanded = pd.concat(expanded_data, ignore_index=True)

    # Interpolate missing values for 'Value' column by crop
    df_expanded['Value'] = df_expanded.groupby('Item')['Value'].transform(lambda x: x.interpolate())

    # Round the 'Value' column to 2 decimal places
    df_expanded['Value'] = df_expanded['Value'].apply(lambda x: round(x, 2))

    # Forward fill for 'Year', 'Months', 'Domain', 'Area', and 'Element'
    df_expanded['Year'] = df_expanded['Year'].fillna(method='ffill')
    df_expanded['Months'] = df_expanded['Months'].fillna(method='ffill')

    # Final cleaning steps
    df_expanded.drop_duplicates(inplace=True)
    df_expanded.dropna(subset=['Date', 'Value'], inplace=True)

    # Sorting by 'Item' and 'Date' for clarity
    df_expanded.sort_values(by=['Item', 'Date'], inplace=True)

    # Remove duplicate and null rows
    df_expanded.reset_index(drop=True, inplace=True)

    merged_df = df_expanded

    

    # Dynamically extract crop options from the "Item" column
    crop_options = merged_df['Item'].unique().tolist()

    # Allow users to select one or more crops
    selected_crops = st.multiselect(
        "Select one or more crops (leave empty to include all crops)",
        options=crop_options,
        default=crop_options  # Pre-select all crops by default
    )

    # Filter data based on selected crops
    if selected_crops:
        df_filtered = merged_df[merged_df['Item'].isin(selected_crops)]
    else:
        df_filtered = merged_df  # If no crops are selected, use the entire dataset
    



    # Visualization type selection
    visualization_type = st.selectbox(
        "Select a type of visualization",
        options=["Price Trends Across All Crops", "Yearly Average Price Trends", "Monthly Average Price Trends", 
                 "Correlation Between Crops", "Histogram of Prices", "Heatmap of Monthly Average Prices",
                 "Pie Chart of Total Prices", "Bar Chart of Monthly Totals"]
    )

    # Generate visualizations based on selected type
    if visualization_type == "Price Trends Across All Crops":
        st.write("### Price Trends Across All Crops")
        pivot_data = df_filtered.pivot_table(
            index='Date',
            columns='Item',
            values='Value',
            aggfunc='mean'
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        for crop in pivot_data.columns:
            ax.plot(pivot_data.index, pivot_data[crop], label=crop)
        ax.set_title("Price Trends Across All Crops")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (LCU)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    elif visualization_type == "Yearly Average Price Trends":
        st.write("### Yearly Average Price Trends for Each Crop")
        yearly_data = df_filtered.groupby(['Year', 'Item'])['Value'].mean().unstack()
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_data.plot(ax=ax)
        ax.set_title("Yearly Average Price Trends")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Price (LCU/tonne)")
        st.pyplot(fig)

    elif visualization_type == "Monthly Average Price Trends":
        st.write("### Monthly Average Price Trends Across All Years")
        months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        monthly_data = df_filtered.groupby(['Months', 'Item'])['Value'].mean().unstack()
        monthly_data = monthly_data.loc[months]  # Ensure months are in correct order
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_data.plot(ax=ax)
        ax.set_title("Monthly Average Price Trends Across All Years")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Price (LCU/tonne)")
        st.pyplot(fig)

    elif visualization_type == "Correlation Between Crops":
        st.write("### Correlation Between Crops")
        crops_data = df_filtered.pivot_table(index='Date', columns='Item', values='Value')
        correlation_matrix = crops_data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Between Crop Prices")
        st.pyplot(fig)

    elif visualization_type == "Histogram of Prices":
        st.write("### Histogram of Prices for Each Crop")
        g = sns.FacetGrid(
            data=df_filtered,
            col="Item",
            col_wrap=1,
            height=6,
            aspect=2,
            sharex=False,
            sharey=False
        )
        g.map(sns.histplot, "Value", kde=True, bins=20, color="blue")
        g.set_titles("{col_name}")
        g.set_axis_labels("Price (LCU/tonne)", "Frequency")
        g.fig.suptitle("Distribution of Producer Prices by Item", y=1.02)
        st.pyplot(g.fig)

    elif visualization_type == "Heatmap of Monthly Average Prices":
        st.write("### Heatmap of Monthly Average Producer Prices")
        monthly_avg = df_filtered.groupby(['Year', 'Months'])['Value'].mean().reset_index()
        monthly_avg_pivot = monthly_avg.pivot_table(index='Year', columns='Months', values='Value', aggfunc='mean')
        scaler = MinMaxScaler()
        monthly_avg_pivot_normalized = pd.DataFrame(
            scaler.fit_transform(monthly_avg_pivot),
            index=monthly_avg_pivot.index,
            columns=monthly_avg_pivot.columns
        )
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        monthly_avg_pivot_normalized = monthly_avg_pivot_normalized[month_order]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            monthly_avg_pivot_normalized,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            cbar=True,
            ax=ax
        )
        ax.set_title("Heatmap of Normalized Monthly Average Producer Prices")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        st.pyplot(fig)

    elif visualization_type == "Pie Chart of Total Prices":
        st.write("### Proportion of Total Producer Prices by Item")
        item_totals = df_filtered.groupby('Item')['Value'].sum()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
        fig, ax = plt.subplots(figsize=(8, 8))
        item_totals.plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(item_totals)],
            wedgeprops={'edgecolor': 'black'},
            ax=ax
        )
        ax.set_title('Proportion of Total Producer Prices by Item')
        ax.set_ylabel('')
        st.pyplot(fig)

    elif visualization_type == "Bar Chart of Monthly Totals":
        st.write("### Monthly Total Producer Prices (All Items)")
        monthly_totals_sum = df_filtered.groupby('Months')['Value'].sum()
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        monthly_totals_sum = monthly_totals_sum.reindex(month_order)
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_totals_sum.plot(kind='bar', ax=ax, color='darkblue', edgecolor='black')
        ax.set_title('Monthly Total Producer Prices (All Items)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Price (LCU/tonne)')
        st.pyplot(fig)

    # Check if at least one of the columns exists
    columns_to_remove = ["Domain", "Area", "Element"]
    existing_columns = [col for col in columns_to_remove if col in df.columns]

    # If any of the columns exist, drop them
    if existing_columns:
        merged_df = merged_df.drop(columns=existing_columns)



    # Save the cleaned data to CSV
    cleaned_csv_filename = "cleaned_crop_data.csv"
    merged_df.to_csv(cleaned_csv_filename, index=False)

    # Provide a download link for the cleaned data
    st.write("### Download the Cleaned Data")
    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=open(cleaned_csv_filename, "rb").read(),
        file_name=cleaned_csv_filename,
        mime="text/csv"
    
    )

    return merged_df
