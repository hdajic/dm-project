from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#####
#KNN
knn = KNeighborsClassifier(n_neighbors=3)
classifier = MultiOutputClassifier(knn, n_jobs=-1)
prediction = knn.fit(X_train, y_train).predict(X_test)
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

sentimentBar = plt.bar(['male', 'female'], [male, female])
plt.ylabel('Number of patients with negative summary')
plt.xlabel('Gender')
plt.title('Number of patients with negative summary based on gender')
sentimentBar[0].set_color('b')
sentimentBar[1].set_color('r')
plt.show()

explode = (0, 0.1, 0, 0)
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
ageBar = plt.pie([old_deaths, adults_deaths, youth_deaths, children_deaths], explode=explode, labels= ['Old', 'Adults', 'Young', 'Children'], 
            colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Number of patients with negative summary based on age')
plt.show()

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
plt.show()


#print(knn.score(X_train, y_train))
