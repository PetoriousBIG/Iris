import matplotlib
import pandas as pd
from matplotlib import pyplot as plt   
import seaborn as sns

dataFrame = pd.read_csv('iris.data', names = ['sepal_length', 'sepal_width',
                                              'petal_length', 'petal_width', 'class'])

print(dataFrame.head())

dataFrame['petal_length'].hist()
plt.tight_layout()
plt.show()

marker_shapes = ['.', '^', '*']

# Then, plot the scatterplot 
ax = plt.axes()
for i, species in enumerate(dataFrame['class'].unique()):
    species_data = dataFrame[dataFrame['class'] == species]
    species_data.plot.scatter(x='petal_length',
                              y='sepal_length',
                              marker=marker_shapes[i],
                              s=100,
                              title="Petal Length vs Sepal Length by Species",
                              label=species, figsize=(10, 7), ax=ax)

plt.show()