import pandas as pd
import numpy

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

my_sheet = 'COVID19_line_list_data'
file_name = 'COVID19-Dataset.xlsx'

df = pd.DataFrame(pd.read_excel(file_name, sheet_name = my_sheet).columns).to_numpy()
df_data = pd.DataFrame(pd.read_excel(file_name, sheet_name = my_sheet)).to_numpy()

#Replacing gender data into number type male = 1, female = 0
for i in range(len(df_data)):
     if df_data[i][5] == 'male':
          df_data[i][5] = 1
     else:
          df_data[i][5] = 0
#############################################################

### Old = 3, Adults = 2, Youth = 1, Children = 0
for i in range(len(df_data)):
     if df_data[i][12] == 'Old':
          df_data[i][12] = 3
     elif df_data[i][12] == 'Adults':
          df_data[i][12]= 2
     elif df_data[i][12] == 'Youth':
          df_data[i][12] = 1
     elif df_data[i][12] == 'Children':
          df_data[i][12] = 0
#################################################

#Deleting unused columns
#########################################################
data = numpy.delete(df_data, [1, 2, 3, 4, 11], axis=1)
df_data = data

columns = numpy.delete(df, [1, 2, 3, 4, 11])
df = columns
#########################################################

print("Data shape = ", df_data.shape)
print("Labels shape = ", df.shape)

#for i in range(len(df_data[0])):
#     print(df_data[0][i])



df_data = df_data.transpose()

X_train, X_test, y_train, y_test = train_test_split(df_data, df)



#################### Knn algorithm ###########################
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#
#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)
#print('Accuracy of K-NN classifier on training set: {:.2f}'
#     .format(knn.score(X_train, y_train)))
##############################################################


################### Random forest ############################
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
#print(predictions)
##############################################################