
#The program contains no error but a few warning which are ignored in consideration of this specific dataset

#Linear Algebra
import numpy as np

#data processing
import pandas as pd

#Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data", names =  ["erythema", "scaling", "definite borders", "itching", "koebner phenomenon", "polygonal papules", "follicular papules", "oral mucosal involvement", "knee and elbow involvement", "scalp involvement", "family history", "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate", "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing of the rete ridges", "thinning of the suprapapillary epidermis", "spongiform pustule", "Age", "class"])

print(df.info())

print(df.describe())

print(df.head(15))

#Dealing with the missing values
print("Missing values")
print((df["Age"] == "?").sum())
        
df[["Age"]] = df[["Age"]].replace("?", np.NaN)
# drop rows with missing values
df.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(df.shape)

print("Missing values")
print((df["Age"] == "?").sum())

#Preparing data for Training

data = df.iloc[:, 0:-1].values
target = df.iloc[:, -1].values

#Spliting the data in test and train data

data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.3, random_state = 0)

#feature Scaling
sc = StandardScaler()
data_train = sc.fit_transform(data_train)
data_test  = sc.fit_transform(data_test)

#Training the model
random_forest = RandomForestClassifier(n_estimators = 3)
random_forest.fit(data_train, target_train)

prediction_forest = random_forest.predict(data_test)



accuracy = round( random_forest.score(data_train, target_train)*100 , 2)
print("Accuracy of Random forest with 3 estimators")
print(round(accuracy, 2), "%")

#Best no. of Estimators
max=0
accu_list = []
for i in range (1,101):
    random_forest = RandomForestClassifier(n_estimators = i)
    random_forest.fit(data_train, target_train)
    prediction_forest = random_forest.predict(data_test)
    accuracy = round( random_forest.score(data_train, target_train)*100 , 2)
    accu_list.append(round(accuracy,2))
    if max < round(accuracy,2):
        max = round(accuracy,2)
        no = i
print("Best no. of Estimators are",no)

import matplotlib.pyplot as plt
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.plot(accu_list,'o-')
plt.xlabel("No. of estimators")
plt.ylabel("Accuracy")

plt.show()


