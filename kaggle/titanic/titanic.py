import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

from sklearn.preprocessing import LabelEncoder

train_x = train_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

for c in ["Sex", "Embarked"]:
    le = LabelEncoder()
    le.fit(train_x[c].fillna("NA"))

    train_x[c] = le.transform(train_x[c].fillna("NA"))
    test_x[c] = le.transform(test_x[c].fillna("NA"))

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)
pred = model.predict_proba(test_x)[:, 1]

import numpy as np
pred_label = np.where(pred > 0.5, 1, 0)


submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission.csv', index=False)
