import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore

data = pd.read_csv('login.csv')

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

plt.plot(range(1, 21), accuracy_scores)
plt.xlabel('Giá trị K')
plt.ylabel('Độ chính xác')
plt.title('Độ chính xác của KNN với các giá trị K khác nhau')
plt.show()