import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from KNN import Knn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

my_sheet = 'COVID19_line_list_data'
file_name = 'COVID19-Dataset.xlsx'

df = pd.DataFrame(pd.read_excel(file_name, sheet_name = my_sheet))

num_of_classes = len(df.age_descriptive.unique())

df['death'] = df['death'].astype(str)
df['ave_sentiment'] = df['ave_sentiment'].astype(str)
df['from Wuhan'] = df['from Wuhan'].astype(str)

X = df.drop(axis=0, columns=['reporting date', 'summary', 'location', 'country', 'symptom'])
Y = np.vstack((df.age_descriptive, df.gender, df.death, df.ave_sentiment, df['from Wuhan'])).T

genderLabels = df['gender'].astype('category').cat.categories.tolist()
ageDescriptiveLabels = df['age_descriptive'].astype('category').cat.categories.tolist()

replace_map_comp_gender = { 'gender' : { k: v for k, v in zip(genderLabels, list(range(1, len(genderLabels) + 1)))}}
replace_map_comp_age_descriptive = { 'age_descriptive' : { k: v for k, v in zip(ageDescriptiveLabels, list(range(1, len(ageDescriptiveLabels) + 1)))}}

X.replace(replace_map_comp_gender, inplace=True)
X.replace(replace_map_comp_age_descriptive, inplace=True)

#print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

age, gender, death,  sentiment, fromWuhan = y_test.T

### RANDOM FOREST ###
#rf = RandomForest(X_train, y_train)

#rf.predict(X_test)
#rf.set_prediction_data()
#rf.plot_num_deaths_per_age()
#rf.plot_num_deaths_per_gender()

#rf.ageScore(age)
#rf.genderScore(gender)
#rf.deathScore(death)
###############################

### KNN ###
knn = Knn(X_train, y_train)

knn.predict(X_test)
knn.set_prediction_data()
#knn.plot_num_patient_neg_summary_based_on_gender()

#knn.ageScore(age)
#knn.genderScore(gender)
#knn.deathScore(death)

knn.crossValidation(X, Y)
###################################

