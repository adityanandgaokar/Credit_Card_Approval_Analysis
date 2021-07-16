

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sn

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

main_df = pd.read_csv('D:/Projects/Credit_Card_Approval/clean.csv')

cat_col = main_df.columns[(main_df.dtypes == 'object').values].tolist()

for i in cat_col:
    labelencoder = LabelEncoder()
    
    main_df[i] = labelencoder.fit_transform(main_df[i])
    
main_df.to_csv('D:/Projects/Credit_Card_Approval/encoded_data.csv', index=False)

features = main_df.drop(['STATUS'], axis = 1)
label = main_df['STATUS']

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 10)




mms = MinMaxScaler()

x_train_scaled = pd.DataFrame(mms.fit_transform(x_train), columns=x_train.columns)
x_test_scaled = pd.DataFrame(mms.fit_transform(x_test), columns=x_test.columns)

smote = SMOTE()

x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)
x_test_smote, y_test_smote = smote.fit_resample(x_test_scaled, y_test)



'implementing logisticregression'
logis_reg_model = LogisticRegression()

logis_reg_model.fit(x_train_smote, y_train_smote)

print('LogisticRegression model accuracy:- ', logis_reg_model.score(x_test, y_test)*100, '%')

'prediction using trained model'
label_predicted = logis_reg_model.predict(x_test_smote)

'confusion matrix'
conf_mat = confusion_matrix(y_test_smote, label_predicted)
print(conf_mat)

plt.figure(figsize=(10,7))
sn.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


'implementing DecisionTreeClassifier'

dec_tree = DecisionTreeClassifier(max_depth=12, min_samples_split=8)

dec_tree.fit(x_train_smote, y_train_smote)

print('Decision Tree Model Accuracy : ', dec_tree.score(x_test_smote, y_test_smote)*100, '%')

'Prediction using trained model'
label_predicted = dec_tree.predict(x_test_smote)

'confusion matrix'
conf_mat = confusion_matrix(y_test_smote, label_predicted)
print(conf_mat)



'xg boost classification'

xgb_model = XGBClassifier()

xgb_model.fit(x_train_smote, y_train_smote)

print('XGB BOOST Model Accuracy : ', xgb_model.score(x_test_smote, y_test_smote)*100, '%')

'Prediction using trained model'
label_predicted = xgb_model.predict(x_test_smote)

'confusion matrix'
conf_mat = confusion_matrix(y_test_smote, label_predicted)
print(conf_mat)




'RandomForestClassifier'
RandomForest_model = RandomForestClassifier(n_estimators=250,
                                            max_depth=12,
                                            min_samples_leaf=16)

RandomForest_model.fit(x_train_smote, y_train_smote)

print('Random Forest Model Accuracy : ', RandomForest_model.score(x_test_smote, y_test_smote)*100, '%')

prediction = RandomForest_model.predict(x_test_smote)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_smote, prediction))
      



'K nearest neighbors'

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 7)

knn_model.fit(x_train_smote, y_train_smote)

print('KNN Model Accuracy : ', knn_model.score(x_test_smote, y_test_smote)*100, '%')

prediction = knn_model.predict(x_test_smote)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_smote, prediction))

























