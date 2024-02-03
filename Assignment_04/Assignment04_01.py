import numpy as np
import pandas as pd

# 1(a) Import the given “Data.csv”
read_Data = pd.read_csv(r'C:\Users\charu\AssignmentNe\Neural-Networks-and-Deep-Learning\Assignment_04\data.csv')
read_Data.info()

#(c) Show the basic statistical description about the data.
read_Data.head()

#(d)Check if the data has null values.
read_Data.isnull().any()

read_Data.fillna(read_Data.mean(), inplace=True)
read_Data.isnull().any()

#d(i)Replace the null values with the mean
column_means = read_Data.mean()
print(column_means)
read_Data = read_Data. fillna(column_means)
print(read_Data.head(20))

#(e)Select at least two columns and aggregate the data using: min, max, count, mean.
res = read_Data.agg({'Calories': ['mean', 'min','max', 'count'],'Pulse': ['mean', 'min', 'max', 'count']})
print(res)

#(f)Filter the dataframe to select the rows with calories values between 500 and 1000
filter_dst_Data1=read_Data[(read_Data['Calories'] > 500) & (read_Data['Calories'] < 1000)]
print(filter_dst_Data1)
#(g)Filter the dataframe to select the rows with calories values > 500 and pulse < 100.
filter_dst_Data2=read_Data[(read_Data['Calories'] > 500) & (read_Data['Pulse'] < 100)]
print(filter_dst_Data2)

#(h)Create a new “df_modified” dataframe that contains all the columns from dst_data except for
#“Maxpulse”.
df_modified = read_Data.loc[:, read_Data.columns != 'Maxpulse']
print(df_modified)

#(i). Delete the “Maxpulse” column from the main dst_data dataframe
read_Data.drop('Maxpulse', inplace=True, axis=1)
print(read_Data.dtypes)

#(j). Convert the datatype of Calories column to int datatype
read_Data["Calories"] = read_Data["Calories"].astype(float).astype(int)
print(read_Data.dtypes)

#(k)Using pandas create a scatter plot for the two columns (Duration and Calories).
import matplotlib.pyplot as plt
as1 = read_Data.plot.scatter(x='Duration',y='Calories')
plt.scatter(x='Duration',y='Calories')
plt.show()
print(as1)




