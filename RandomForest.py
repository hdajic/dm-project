from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class RandomForest:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        rf = RandomForestClassifier(n_estimators=20, random_state=1)
        self.multi_target_forest = MultiOutputClassifier(rf, n_jobs=-1)

    def predict(self, x_test):
        self.x_test = x_test
        self.predictions = self.multi_target_forest.fit(self.x_train, self.y_train).predict(x_test)

    def set_prediction_data(self):
        self.age_axis, self.gender_axis, self.death_axis, self.sentiment_axis, self.fromWuhan_axis = self.predictions.T
        self.sentiment_axis = self.sentiment_axis.astype(float)

    def plot_num_deaths_per_gender(self):
        male_deaths = 0
        female_deaths = 0
        for i in range(len(self.gender_axis)):
            if self.gender_axis[i] == 'male':
                male_deaths += int(self.death_axis[i])
            else:
                female_deaths += int(self.death_axis[i])

        gender = ['male', 'female']
        num_of_deaths = [male_deaths, female_deaths]
        genderBar = plt.bar(gender, num_of_deaths)
        plt.ylabel('Number of deaths')
        plt.xlabel('Gender')
        genderBar[0].set_color('b')
        genderBar[1].set_color('r')
        plt.title('Number of deaths per gender')
        plt.show()

    def plot_num_deaths_per_age(self):
        adults_deaths = 0
        old_deaths = 0
        children_deaths = 0
        youth_deaths = 0

        for i in range(len(self.age_axis)):
            if self.age_axis[i] == 'Adults':
                adults_deaths += int(self.death_axis[i])
            elif self.age_axis[i] == 'Old':
                old_deaths += int(self.death_axis[i])
            elif self.age_axis[i] == 'Children':
                children_deaths += int(self.death_axis[i])
            elif self.age_axis[i] == 'Youth':
                youth_deaths += int(self.death_axis[i])

        age_descriptive = ['Children', 'Youth', 'Adults', 'Old']
        deaths_per_age = [children_deaths, youth_deaths, adults_deaths, old_deaths]

        ageDescriptiveBar = plt.bar(age_descriptive, deaths_per_age)
        plt.ylabel('Number of deaths')
        plt.xlabel('Age descriptive')
        plt.title('Number of deaths per age')
        ageDescriptiveBar[0].set_color('b')
        ageDescriptiveBar[1].set_color('r')
        ageDescriptiveBar[2].set_color('g')
        ageDescriptiveBar[2].set_color('y')
        plt.show()

    def writeTestDataIntoCsvFile(self, fileName):
        path = Path(__file__).parent.absolute().joinpath(fileName + '.csv')
        self.x_test.to_csv(path)
    
    def writePredictionIntoFile(self, fileName):
        fileName+='.txt'
        predictionFile = open(fileName, "w+")
        for i in range(len(self.predictions)):
            for j in range(len(self.predictions[i])):
                predictionFile.write(self.predictions[i][j] + "  ")
            predictionFile.write('\r\n')
        predictionFile.close()

    def getPrediction(self):
        return self.predictions

    def ageScore(self, ageData):
        print('Age score confusion matrix: ', confusion_matrix(ageData,self.age_axis))
        print('Age score classification report: ', classification_report(ageData,self.age_axis))
        print('Age accuracy score: ', accuracy_score(ageData, self.age_axis))

    def genderScore(self, genderData):
        print('Gender score confusion matrix: ', confusion_matrix(genderData,self.gender_axis))
        print('Gender score classification report: ', classification_report(genderData,self.gender_axis))
        print('Gender accuracy score: ', accuracy_score(genderData, self.gender_axis))

    def deathScore(self, deathData):
        print('Death score confusion matrix: \n', confusion_matrix(deathData,self.death_axis))
        print('Deaht score classification report: \n', classification_report(deathData,self.death_axis))
        print('Deaht accuracy score: \n', accuracy_score(deathData, self.death_axis))