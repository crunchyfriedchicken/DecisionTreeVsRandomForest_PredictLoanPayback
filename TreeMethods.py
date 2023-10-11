# Import packages -------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get data --------------------------------------------------------------
df = pd.read_csv("./data/loan_data.csv")
df.head()
df.info()
df.describe()

# Exploratory data analysis ---------------------------------------------
sns.set_style("whitegrid")
#sns.pairplot(df, hue="not.fully.paid")
# I want to check out relationship between not.fully.paid vs FICO and int.rate
fico1 = df["fico"].loc[df["not.fully.paid"]==1]
fico0 = df["fico"].loc[df["not.fully.paid"]==0]
sns.histplot(fico0,label="Not Fully Paid = 0")
sns.histplot(fico1,label="Not Fully Paid = 1")
plt.legend()
plt.show()

sns.jointplot(df,x= "fico",y="int.rate")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(df,x="purpose",hue="not.fully.paid")
plt.show()

sns.lmplot(df, x="fico",y="int.rate",hue="credit.policy",col="not.fully.paid")
plt.show()

# Clean data [make dummy variables and remove text columns] -------------
# change purpose into dummy variable
df.info()
# 14 columns x 9578 rows
new_df = pd.get_dummies(df, columns =["purpose"],drop_first=True)
len(df["purpose"].unique())
# 7 
# check: 19 columns x 9578 rows -> 14 col + (7-1) - 1 = 14

# Check for nulls -------------------------------------------------------
sns.heatmap(new_df.isnull(),yticklabels=False,cbar=False)
plt.show()

# Check out correlation between variables [optional]---------------------
sns.heatmap(new_df.corr(),annot=True) 
plt.show()

# Splitting data --------------------------------------------------------
X = new_df.drop("not.fully.paid",axis=1)
y = new_df["not.fully.paid"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=101,test_size=0.3)

# Train decision tree model & get predictions ---------------------------
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dpredictions = dtree.predict(X_test)

# Train random forest model & get predictions ---------------------------
from sklearn.ensemble import RandomForestClassifier
rtree = RandomForestClassifier()
rtree.fit(X_train,y_train)
rpredictions = rtree.predict(X_test)

# Compare results -------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report
# Decision tree results:
print("Decision tree results:")
print(confusion_matrix(y_test,dpredictions))
print(classification_report(y_test,dpredictions))
print("\n")
# Random forest results:
print("Random forest results:")
print(confusion_matrix(y_test,rpredictions))
print(classification_report(y_test,rpredictions))

# Evaluate model --------------------------------------------------------
"""
- RF has higher accuracy at 0.85 compared to DT at 0.73.
- For class 0: RF has higher F1 score, recall and same precision
- For class 1: Both struggle with recall and F1 score with DT performing better for those and RF performing better for precision

Overall, RF performs better. 
"""
