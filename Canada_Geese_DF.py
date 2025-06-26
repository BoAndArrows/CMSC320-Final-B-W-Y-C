import pandas as pd
from IPython.display import HTML, Image

canada_geese_df = pd.read_csv("Bird_Banding_Data/NABBP_2023_grp_02.csv") 

subspecies_df = pd.read_csv("Bird_Banding_Data/NABBP_2023_grp_03.csv")

print(canada_geese_df)