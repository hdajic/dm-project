from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


class Knn:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.knn = KNeighborsClassifier(n_neighbors=3)

    def predict(self, x_test):
            self.prediction = self.knn.fit(self.x_train, self.y_train).predict(x_test)

    def set_prediction_data(self):
        self.age_axis, self.gender_axis, self.death_axis, self.sentiment_axis, self.fromWuhan_axis = self.prediction.T
        self.sentiment_axis = self.sentiment_axis.astype(float)
    
    def plot_num_patient_neg_summary_based_on_gender(self):
        gender = []
        for i in range(len(self.sentiment_axis)):
            if self.sentiment_axis[i] < 0.000001:
                gender.append(self.gender_axis[i]) 
        male = 0
        female = 0

        for i in range(len(gender)):
            if gender[i] == 'male':
                male+=1
            else:
                female+=1
        sentimentBar = plt.bar(['male', 'female'], [male, female])
        plt.ylabel('Number of patients with negative summary')
        plt.xlabel('Gender')
        plt.title('Number of patients with negative summary based on gender')
        sentimentBar[0].set_color('b')
        sentimentBar[1].set_color('r')
        plt.show()

    def plot_num_patient_neg_summary_baseg_on_age(self):
        age = []
        for i in range(len(self.sentiment_axis)):
            if self.sentiment_axis[i] < 0.000001:
                age.append(self.age_axis[i])
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
        explode = (0, 0.1, 0, 0)
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
        plt.pie([old_deaths, adults_deaths, youth_deaths, children_deaths], explode=explode, labels= ['Old', 'Adults', 'Young', 'Children'], 
            colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Number of patients with negative summary based on age')
        plt.show()

    def plot_num_patient_neg_summary_based_on_is_from_wuhan(self):
        isFromWuhan = 0
        notFromWuhan = 0
        fromWuhan = []
        
        for i in range(len(self.sentiment_axis)):
            if self.sentiment_axis[i] < 0.000001:
                fromWuhan.append(self.fromWuhan_axis[i])
        
        for i in range(len(fromWuhan)):
            if fromWuhan[i] == '1':
                isFromWuhan += 1
            else:
                notFromWuhan += 1

        explode = (0, 0.1)
        colors = ['gold', 'yellowgreen']
        plt.pie([isFromWuhan, notFromWuhan], explode=explode, labels= ['From Wuhan', 'Not from Wuhan'], 
                    colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Number of patients with negative summary')
        plt.show()