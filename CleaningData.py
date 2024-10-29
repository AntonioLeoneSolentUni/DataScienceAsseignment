import pandas as pd

dataframe = pd.read_csv('MedicalDatasetVision.csv')

print(dataframe)

#Remove Empty Cells
new_DF = dataframe.fillna("No")
print(new_DF)

#Removing Duplicates
new_DF = dataframe.drop_duplicates()
print(new_DF)

# Removing Wrong Data and applieng rules:
# Rule:
#