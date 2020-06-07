import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import tree
df = pd.read_csv("car.csv")
df.head(13)
inputs = df.drop('car_problem' ,axis='columns')
target = df['car_problem']
target
le_Symptom_1 = LabelEncoder()
le_symptom_2 = LabelEncoder()
le_symptom_3 = LabelEncoder()
inputs['Symptom_1_n'] = le_Symptom_1.fit_transform(inputs['Symptom 1'].astype(str))
inputs['Symptom_2_n'] = le_symptom_2.fit_transform(inputs['symptom 2'].astype(str))
inputs['Symptom_3_n'] = le_symptom_3.fit_transform(inputs['symptom 3'].astype(str))
inputs.head(13)
inputs_n = inputs.drop(['Symptom 1','symptom 2','symptom 3'] ,axis = 'columns')
inputs_n
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

model
print("for example 1 , 2, 3")
print("Describe 3 symptoms to your car problem")

val1 = input("symptom 1:")
val2 = input("symptom 2:")
val3 = input("symptom 3:")
#Val 1, 2 and 3 can be determined from the table above
#for example 1 , 2 , 4 for first problem
print(val1,val2,val3)
model.predict([[val1, val2, val3]])