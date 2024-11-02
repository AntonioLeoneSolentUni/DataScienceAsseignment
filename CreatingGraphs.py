import pandas as pd
import matplotlib.pyplot as mp

df = pd.read_csv('MedicalDatasetVision.csv')

for item in df['Sample_Size']:
    test = item