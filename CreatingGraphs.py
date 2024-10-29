import pandas as pd
import matplotlib.pyplot as mp

dataframe = pd.read_csv('MedicalDatasetVision.csv')
amountOfAsian = dataframe['RaceEthnicityID']



#amountOfAsian = dataframe[dataframe["RaceEthnicity"] == "Asian"].count()


mp.hist(dataframe["Age"])
mp.show()