import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.neighbors import KNeighborsClassifier


my_sheet = 'COVID19_line_list_data'
file_name = 'COVID19-Dataset.xlsx'

df = pd.DataFrame(pd.read_excel(file_name, sheet_name = my_sheet))

#print(df.head())

#print(df.shape)

num_of_classes = len(df.age_descriptive.unique())

print(df.describe())

#print(num_of_classes)

X = df.drop(axis=0, columns=['reporting date', 'summary', 'location', 'country', 'symptom'])
#Y = df.age_descriptive
df['death'] = df['death'].astype(str)
Y = np.vstack((df.age_descriptive, df.gender, df.death)).T

print(X.shape)
print(Y.shape)

genderLabels = df['gender'].astype('category').cat.categories.tolist()
ageDescriptiveLabels = df['age_descriptive'].astype('category').cat.categories.tolist()

replace_map_comp_gender = { 'gender' : { k: v for k, v in zip(genderLabels, list(range(1, len(genderLabels) + 1)))}}
replace_map_comp_age_descriptive = { 'age_descriptive' : { k: v for k, v in zip(ageDescriptiveLabels, list(range(1, len(ageDescriptiveLabels) + 1)))}}

X.replace(replace_map_comp_gender, inplace=True)
X.replace(replace_map_comp_age_descriptive, inplace=True)

#print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#RF
rf = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(rf, n_jobs=-1)
predictions = multi_target_forest.fit(X_train, y_train).predict(X_test)
#print(predictions)

x_axis, y_axis, z_axis = predictions.T

death_sum = sum(1 for y in z_axis if y == '1')

plt.bar(x_axis, death_sum)
plt.show()


#####

#KNN
#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)
#print('Accuracy of K-NN classifier on training set: {:.2f}'
#     .format(knn.score(X_train, y_train)))
######