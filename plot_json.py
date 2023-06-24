# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import urllib.request

home = '/Users/rcvb/Documents/tcc_rian/code'
with open(f'{home}/assets/confirmed_cases_by_region_and_date.json') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df.reset_index(inplace=True)
df.rename(columns={'index':'collect_date'}, inplace=True)
df['collect_date'] = pd.to_datetime(df['collect_date'])
df.sort_values(by=['collect_date'], inplace=True)
df.fillna(0, inplace=True)

# Restructure the dataframe
df_melted = df.melt(id_vars='collect_date', var_name='region', value_name='cases')

# Ensure data is sorted by date
df_melted['collect_date'] = pd.to_datetime(df_melted['collect_date'])
df_melted = df_melted.sort_values('collect_date')

# Set a style for seaborn
sns.set(style="whitegrid")

# Creating a figure with size 10x10
plt.figure(figsize=(10, 10))

# Plotting a lineplot with seaborn
sns.lineplot(data=df_melted, x="collect_date", y="cases", hue="region")

# Setting the title and labels
plt.title("COVID-19 Casos Confirmados por Região em SC nos últimos 15 dias")
plt.xlabel("Data")
plt.ylabel("Número de Casos Confirmados últimos 15 dias")

# Display the plot
plt.show()
