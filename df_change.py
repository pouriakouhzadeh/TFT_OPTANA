import pandas as pd

df = pd.read_csv("new_calculation_7_years.csv")
# df1 = pd.read_csv("filtered_df_7_years.csv")
# df.drop(["Sequence_1", "Sequence_2", "Sequence_3", "Sequence_5", "Sequence_20", "Sequence_30", "Sequence_40", "Sequence_60"], axis=1, inplace=True)



# Extract the minute part from the "Time of Day" column
df['Minutes'] = pd.to_datetime(df['Time of Day']).dt.minute

# Filter the dataset to keep rows where the minutes are a multiple of 10 (i.e., 00, 10, 20, 30, 40, 50)
filtered_df = df[df['Minutes'] % 10 == 0]

# Drop the 'Minutes' column as it is no longer needed
filtered_df = filtered_df.drop(columns=['Minutes'])

# Save the filtered data to a new CSV file
filtered_df.to_csv('filtered_data.csv', index=False)

# Display the filtered data
print(filtered_df.head())







print(df)
filtered_df.to_csv("filtered_df_7_years_.csv")