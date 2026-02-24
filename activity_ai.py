import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Separate Features and Target
X_train = train.drop(['Activity', 'subject'], axis=1)
y_train = train['Activity']
X_test = test.drop(['Activity', 'subject'], axis=1)
y_test = test['Activity']

# 3. Train Model
# LightGBM is chosen because it's very efficient for low-power devices (like phones)
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print(f"✅ Recognition Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 5. Visualizing the Confusion Matrix
# This shows us if the model confuses 'Walking' with 'Walking Upstairs'
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Activity Recognition Results')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
