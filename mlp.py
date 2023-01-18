

# Biblioteca Pandas
import io, os
from sklearn.model_selection import train_test_split # Import train_test_split function
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt


path = os.path.abspath(".")

data = pd.read_csv('./autism_dataset.csv', encoding='utf-8')


df = pd.DataFrame(data)

df.drop(['age', 'relation', 'contry_of_res', 'used_app_before', 'age_desc', 'ethnicity', 'Class/ASD'],axis=1, inplace=True)
print(df)
austim = { 'yes':1 ,'no': 0}
jaundice = { 'yes':1 ,'no': 0}
gender = {'m':1, 'f':0}
df['austim'] = df['austim'].map(austim)
df['jaundice'] = df['jaundice'].map(jaundice)
df['gender'] = df['gender'].map(gender)
df = df.rename(columns = {'austim': 'autism_familly'})

feature_cols = list(df.columns)

X = df[feature_cols] # Features
y = data['Class/ASD'] # Target variable


# Split dataset into training set and test set - 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5) 

mlp = MLPClassifier(activation='relu')
mlp.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = mlp.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Acurácia 70/30: %.3f' % metrics.accuracy_score(y_test, y_pred))
print("Matriz de confusão para 70/30:")
print(confusion_matrix(y_test, y_pred))
#-------------------------------------------------------------------------

# Split dataset into training set and test set - 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5) 

mlp = MLPClassifier(activation='relu')
mlp.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = mlp.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Acurácia 80/20: %.3f' % metrics.accuracy_score(y_test, y_pred))
print("Matriz de confusão para 80/20:")
print(confusion_matrix(y_test, y_pred))

#-------------------------------------------------------------------------

# Split dataset into training set and test set - 90% training and 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5) 

mlp = MLPClassifier(activation='relu')
mlp.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = mlp.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Acurácia 90/10: %.3f' % metrics.accuracy_score(y_test, y_pred))
print("Matriz de comfusão para 90/10:")
print(confusion_matrix(y_test, y_pred))

#gerar graficos para analise
n_cols = 799
n_classes = 2 
A = round((n_cols + n_classes)/2)
O = 2
AO = round((A+O)/2)

eixo_x = [O, AO, A]
eixo_y = []
for it in eixo_x:
    print(it)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5) 

    mlp = MLPClassifier(hidden_layer_sizes=it, activation='relu', solver='adam')
    mlp.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = mlp.predict(X_test)
    eixo_y.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(eixo_x, eixo_y)
plt.ylabel("Acurácia")
plt.xlabel("Número de neurônios")
plt.show()
