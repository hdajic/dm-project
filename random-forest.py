from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt


#RF
rf = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(rf, n_jobs=-1)
predictions = multi_target_forest.fit(X_train, y_train).predict(X_test)
print(predictions)

x_axis, y_axis, z_axis, k_axis = predictions.T

print(k_axis)

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
plt.show()

