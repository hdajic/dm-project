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

#print(df.describe())

#print(num_of_classes)

X = df.drop(axis=0, columns=['reporting date', 'summary', 'location', 'country', 'symptom'])
#Y = df.age_descriptive
df['death'] = df['death'].astype(str)
df['ave_sentiment'] = df['ave_sentiment'].astype(str)
df['from Wuhan'] = df['from Wuhan'].astype(str)
Y = np.vstack((df.age_descriptive, df.gender, df.death, df.ave_sentiment, df['from Wuhan'])).T

#print(X.shape)
#print(Y.shape)

genderLabels = df['gender'].astype('category').cat.categories.tolist()
ageDescriptiveLabels = df['age_descriptive'].astype('category').cat.categories.tolist()

replace_map_comp_gender = { 'gender' : { k: v for k, v in zip(genderLabels, list(range(1, len(genderLabels) + 1)))}}
replace_map_comp_age_descriptive = { 'age_descriptive' : { k: v for k, v in zip(ageDescriptiveLabels, list(range(1, len(ageDescriptiveLabels) + 1)))}}

X.replace(replace_map_comp_gender, inplace=True)
X.replace(replace_map_comp_age_descriptive, inplace=True)

#print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#RF
#rf = RandomForestClassifier(random_state=1)
#multi_target_forest = MultiOutputClassifier(rf, n_jobs=-1)
#predictions = multi_target_forest.fit(X_train, y_train).predict(X_test)
#print(predictions)

#x_axis, y_axis, z_axis, k_axis = predictions.T

#print(k_axis)
'''
male_deaths = 0
female_deaths = 0
for i in range(len(y_axis)):
    if y_axis[i] == 'male':
        male_deaths += int(z_axis[i])
    else:
        female_deaths += int(z_axis[i])

death_sum = sum(1 for y in z_axis if y == '1')

##### NUMBER OF DEATHS PER GENDER ##############
gender = ['male', 'female']
num_of_deaths = [male_deaths, female_deaths]
genderBar = plt.bar(gender, num_of_deaths)
plt.ylabel('Number of deaths')
plt.xlabel('Gender')
genderBar[0].set_color('b')
genderBar[1].set_color('r')
plt.title('Number of deaths per gender')
#plt.show()
################################################

#############################################
plt.clf() #Clears a figure

adults_deaths = 0
old_deaths = 0
children_deaths = 0
youth_deaths = 0


for i in range(len(x_axis)):
    if x_axis[i] == 'Adults':
        adults_deaths += int(z_axis[i])
    elif x_axis[i] == 'Old':
        old_deaths += int(z_axis[i])
    elif x_axis[i] == 'Children':
        children_deaths += int(z_axis[i])
    elif x_axis[i] == 'Youth':
        youth_deaths += int(z_axis[i])

age_descriptive = ['Youth', 'Children', 'Adults', 'Old']
deaths_per_age = [youth_deaths, children_deaths, adults_deaths, old_deaths]

ageDescriptiveBar = plt.bar(age_descriptive, deaths_per_age)
plt.ylabel('Number of deaths')
plt.xlabel('Age descriptive')
plt.title('Number of deaths per age')
ageDescriptiveBar[0].set_color('b')
ageDescriptiveBar[1].set_color('r')
ageDescriptiveBar[2].set_color('g')
ageDescriptiveBar[2].set_color('y')
#plt.show()

#############################################



#############################################
'''


#####
#KNN
knn = KNeighborsClassifier(n_neighbors=3)
classifier = MultiOutputClassifier(knn, n_jobs=-1)
prediction = classifier.fit(X_train, y_train).predict(X_test)
print(prediction)
#print('Accuracy of K-NN classifier on training set: {:.2f}'
#     .format(knn.score(X_train, y_train)))
######

age_axis, gender_axis, death_axis, sentiment_axis, fromWuhan_axis = prediction.T

gender = []
age = []
sentiment_axis = sentiment_axis.astype(float)
fromWuhan = []
for i in range(len(sentiment_axis)):
    if sentiment_axis[i] < 0.000001:
        gender.append(gender_axis[i]) 
        age.append(age_axis[i])
        fromWuhan.append(fromWuhan_axis[i])

male = 0
female = 0

for i in range(len(gender)):
    if gender[i] == 'male':
        male+=1
    else:
        female+=1

adults_deaths = 0
old_deaths = 0
children_deaths = 0
youth_deaths = 0
for i in range(len(age)):
    if age[i] == 'Adults':
        adults_deaths += 1
    elif age[i] == 'Old':
        old_deaths += 1
    elif age[i] == 'Children':
        children_deaths += 1
    elif age[i] == 'Youth':
        youth_deaths += 1

'''
sentimentBar = plt.bar(['male', 'female'], [male, female])
plt.ylabel('Number of patients with negative summary')
plt.xlabel('Gender')
plt.title('Number of patients with negative summary based on gender')
sentimentBar[0].set_color('b')
sentimentBar[1].set_color('r')
plt.show()
'''

##############
'''
explode = (0, 0.1, 0, 0)
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
ageBar = plt.pie([old_deaths, adults_deaths, youth_deaths, children_deaths], explode=explode, labels= ['Old', 'Adults', 'Young', 'Children'], 
            colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Number of patients with negative summary based on age')
plt.show()
'''
##############

isFromWuhan = 0
notFromWuhan = 0

for i in range(len(fromWuhan)):
    if fromWuhan[i] == '1':
        isFromWuhan+=1
    else:
        notFromWuhan+=1


explode = (0, 0.1)
colors = ['gold', 'yellowgreen']
ageBar = plt.pie([isFromWuhan, notFromWuhan], explode=explode, labels= ['From Wuhan', 'Not from Wuhan'], 
            colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Number of patients with negative summary ')
#plt.show()


print(format(knn.score(X_train, y_train)))


##score i klasterizacija
