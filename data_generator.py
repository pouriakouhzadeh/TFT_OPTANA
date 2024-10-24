import pandas as pd
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("new_calculation_7_years.csv")

# Extract the minute part from the "Time of Day" column
df['Minutes'] = pd.to_datetime(df['Time of Day']).dt.minute

# Create a new DataFrame to store the updated rows
aggregated_df = pd.DataFrame()

# Use tqdm to add progress bar for the iteration
for i in tqdm(range(len(df)), desc="Processing rows"):
    # Check if the minute is a multiple of 10 (00, 10, 20, 30, 40, 50)
    if df.loc[i, 'Minutes'] % 10 == 0:
        # Get the previous 9 rows (if available)
        start_index = max(0, i - 9)
        prev_rows = df.iloc[start_index:i]

        # Sum the numeric values of previous rows and add to the current row
        summed_row = prev_rows.select_dtypes(include='number').sum()
        for col in summed_row.index:
            df.at[i, col] += summed_row[col]
        
        # Append the updated row to the new DataFrame using _append (new method)
        aggregated_df = aggregated_df._append(df.iloc[i], ignore_index=True)

# Drop the 'Minutes' column as it is no longer needed
aggregated_df = aggregated_df.drop(columns=['Minutes'])

# Save the filtered and aggregated data to a new CSV file
aggregated_df.to_csv('aggregated_filtered_data.csv', index=False)

# Display the updated data
print(aggregated_df.head())
