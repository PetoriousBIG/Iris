import matplotlib
matplotlib.use("TkAgg")
from utils import preprocess
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(16)

try:
    df = pd.read_csv('iris.data', names = ['sepal_length', 'sepal_width',
                                            'petal_length', 'petal_width', 'class'])
except:
    print("""
      Error: File not found
      """)
    quit()

df = preprocess(df)
X = df.loc[:, df.columns[:-3]]
y = df.loc[:, df.columns[-3:]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(4, )))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, verbose=False, shuffle=True)

# Results - Accuracy
scores = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

# Results - Confusion Matrix
y_test_pred = model.predict_classes(X_test)
flat_y_test = y_test.values.argmax(axis=1)

c_matrix = confusion_matrix(flat_y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                 yticklabels=['Setosa', 'Versicolor', 'Virginica'], cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()

