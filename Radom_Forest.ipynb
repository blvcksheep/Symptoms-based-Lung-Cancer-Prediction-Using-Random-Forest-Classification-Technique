#Importing the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, confusion_matrix ,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette = 'deep', rc = {'axes.grid':True})
%matplotlib inline

#Loading the data
data = pd.read_csv("sample_data/survey lung cancer.csv")
data.head()

data.info()

data_new = data.drop(['GENDER','AGE', 'SMOKING', 'ALCOHOL CONSUMING', 'CHRONIC DISEASE', 'PEER_PRESSURE', 'ALLERGY '], axis = 1)
symptoms = [ 'YELLOW_FINGERS', 'ANXIETY', 'FATIGUE ',  'WHEEZING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
X = data_new[symptoms]
y = data_new.LUNG_CANCER
X.head()
y.head()

#Loading the data: Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state = 42, stratify = y)

#Loading the data: Checking for missing entries
X_train.isnull().sum()

#Data Analysis: Convert the target column into a numeric value using Sklearn's LabelEncoder
le = LabelEncoder()
y_train= le.fit_transform(y_train)
y_test= le.transform(y_test)

key = {2: 'yes', 1: 'no'}
for sys in symptoms:
	sns.countplot(x = X_train[sys].replace(key))    
	plt.show()
  
sns.set(style = 'darkgrid',palette = 'bright')
sns.countplot(x = pd.Series(y_train).replace([0,1],['No','Yes']))

# Model Building: Using random forest model
model =  RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('The accuracy score of this Random Forest Classifier model is: {0:.1f}%'.format(100*accuracy_score(y_test, y_pred)))

sns.set(rc = {'axes.grid':False})
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion matrix of the model')
plt.show()

print('The precision of this model is {0:.1f}%'.format(100* precision_score(y_test,y_pred)))

print ('The recall score of this model is {0:.1f}%'.format(100*recall_score(y_test, y_pred)))

print('The harmonic mean of the precision score and recall score is:', f1_score(y_test,y_pred))


# Model Building: Features Importance
Symptoms_importance = pd.DataFrame(  {"Symptoms": list(X.columns), "importance": model.feature_importances_}).sort_values("importance", ascending=False)
# Display
print(Symptoms_importance)

# Model Building: Creatig a Bar Plot
sns.set(palette = 'bright',rc ={'axes.grid':True})
sns.barplot(x=Symptoms_importance.Symptoms, y=Symptoms_importance.importance)
plt.xlabel("Symptoms ")
plt.ylabel("Importance score")
plt.title("The symptoms and their importance in building this model")
plt.xticks( rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
plt.show()










