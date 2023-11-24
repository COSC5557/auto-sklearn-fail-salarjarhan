import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))

from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(
    time_left_for_this_task=300, #60 reaches to the comparable result.
    per_run_time_limit=10,
    resampling_strategy='cv',
    seed=42
    )
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))


'''
The decline in performance can be due to allowing the random forest to grow excessively deep within a 5-minute timeframe. This leads to overfitting on the training data and poor performance on the test data. To overcome this issue, limiting the runtime and using additional techniques such as cross-validation (CV) can effectively enhance the performance.

RF Accuracy 0.6525
AutoML Accuracy 0.6625

References
https://chat.openai.com/share/c79db870-f740-4e45-8464-ebfef97e1f6b
https://stackoverflow.com/questions/46903621/why-my-random-forest-gets-worse-performance-than-decision-tree
'''

