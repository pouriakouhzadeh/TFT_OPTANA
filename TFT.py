import pandas as pd

df = pd.read_csv("new_calculation_three_days.csv")
df.drop(["Sequence_1", "Sequence_2", "Sequence_3", "Sequence_5","Sequence_20","Sequence_30","Sequence_40","Sequence_60"], axis=1, inplace=True)

print(df)

df.to_csv("df.csv")