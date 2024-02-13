import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

#The glass data is loaded from the 'glass.csv' file into a DataFrame 
glass_data = pd.read_csv(r"C:\Users\charu\AssignmentNe\Neural-Networks-and-Deep-Learning\Assignment_05\glass.csv")

x_train = glass_data.drop("Type", axis=1)
y_train = glass_data['Type']


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# Train the model using the training sets
gnb = GaussianNB()
gnb.fit(x_train, y_train)


y_pred = gnb.predict(x_test)
# Classification report 
qual_report = classification_report(y_test, y_pred)
print(qual_report)
print("Naive Bayes accuracy is: ",  (accuracy_score(y_test, y_pred))*100)