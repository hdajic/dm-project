import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from KNN import Knn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

my_sheet = 'COVID19_line_list_data'
file_name = 'COVID19-Dataset.xlsx'

df = pd.DataFrame(pd.read_excel(file_name, sheet_name = my_sheet))

original_df = pd.DataFrame.copy(df)

num_of_classes = len(df.age_descriptive.unique())

df['death'] = df['death'].astype(str)
df['ave_sentiment'] = df['ave_sentiment'].astype(str)
df['from Wuhan'] = df['from Wuhan'].astype(str)

X = df.drop(axis=0, columns=['reporting date', 'summary', 'location', 'country', 'symptom', 'death'])
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
rf = RandomForest(X_train, y_train)

rf.predict(X_test)
rf.set_prediction_data()
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
#knn.plot_num_patient_neg_summary_baseg_on_age()
#knn.plot_num_patient_neg_summary_based_on_is_from_wuhan()

#knn.ageScore(age)
#knn.genderScore(gender)
#knn.deathScore(death)
###################################

T_1 = df.drop(columns=['reporting date', 'summary', 'location', 'country', 'symptom', 'death'])
T_1.replace(replace_map_comp_gender, inplace=True)
T_1.replace(replace_map_comp_age_descriptive, inplace=True)

T = np.array(T_1)
T = preprocessing.scale(T)
Z = np.array(df['death'])

##### Kmeans clustering ###########
model = KMeans(n_clusters=2)
model.fit(T)
correct = 0
for i in range(len(T)):
    predict_me = np.array(T[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = model.predict(predict_me)
    if prediction[0] == int(Z[i]):
        correct += 1

print(correct/len(T))


############### Mean Shift clustering ###############################
'''
X = np.vstack((X_test.age, X_test.ave_sentiment.astype(float))).T
model = MeanShift()
f = model.fit_predict(T)
'''
'''
for cluster in clusters:
    row_ix = np.where(f1 == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1])

plt.xlabel('Person age')
plt.ylabel('Average sentiment')
plt.title('Diagnosis in function of the age')
plt.show()
'''

labels = model.labels_
cluster_centers = model.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(T)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

death_rates = {}

for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']== float(i))]
    death_cluster = temp_df[(temp_df['death']==1)]
    death_rate = len(death_cluster)/len(temp_df)
    death_rates[i] = death_rate
print('Death rate: ', death_rates)

