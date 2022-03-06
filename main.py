# February 2022
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import export_graphviz

warnings.filterwarnings('ignore')

def fetch_student_data():
    # Store student data in csv in dict
    student_data_dict = pd.read_csv('student-mat.csv', sep=';')
    del student_data_dict['G1']
    del student_data_dict['G2']
    return student_data_dict

def show_histograms(student_data_dict):
    # Show historgrams of all attributes
    student_data_dict.hist(figsize=(10,10))
    plt.show()

def find_correlations(student_data_dict):
    # Find correlations amoung numerical attributes with respect to 'G3'
    corr_matrix = student_data_dict.corr()
    top_three = {}

    # Results
    print("Top Three Strongly Correlated Numerical Attributes")
    print("Attributes   Correlation")
    print(corr_matrix['G3'].sort_values(ascending=False).drop('G3')[:3])
    print("\nAll Correlated Numerical Attributes")
    print("Attributes    Correlation")
    print(corr_matrix['G3'].sort_values(ascending=False).drop('G3'))

def transform_dataset(student_data_dict):
    # Transform categorical features and scale
    features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    label_encoder = LabelEncoder()
    hot_encoder = OneHotEncoder()

    for feature in features:
        targets = np.array(student_data_dict[feature])
        new_target = label_encoder.fit_transform(targets)
        new_hot_targets = hot_encoder.fit_transform(new_target.reshape(-1, 1))
        student_data_dict[feature] = new_hot_targets.toarray()

    target = 'G3'
    X = np.array(student_data_dict.drop([target], 1))
    Y = np.array(student_data_dict[target])

    return X, Y

def split_dataset(student_data_dict):
    # Split into training and testing datasets
    X, Y = transform_dataset(student_data_dict)

    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X_tr, X_ts, Y_tr, Y_ts

def most_important_features(model, features, n):
    # Get top n number of important features in model
    important_features = dict(zip(features, model.feature_importances_))
    important_features = sorted(important_features.items(), key=lambda f: f[1], reverse=True)
    
    # Results
    print("\nMost Important Features")
    for feature, score in important_features[:n]:
        print("Feature: %s, Score: %.5f" %(feature, score))

def Decision_Tree_Regressor(X_tr, X_ts, Y_tr, Y_ts):
    # Decision Tree train and test
    regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
    regressor.fit(X_tr, Y_tr)
    train_predict = regressor.predict(X_tr)
    test_predict = regressor.predict(X_ts)
    train_error_calc = mean_squared_error(Y_tr, train_predict)
    test_error_calc = mean_squared_error(Y_ts, test_predict)

    # Results
    print("\nDecision Tree Regressor")
    print("------------------------------------")
    print("Training Mean Squared Error: %.3f" %(train_error_calc))
    print("Testing Mean Squared Error: %.3f" %(test_error_calc))
    return regressor

def Ensemble(X_tr, X_ts, Y_tr, Y_ts):
    # Ensemble train and test to increase prediction
    ensemble = GradientBoostingRegressor(random_state=42)
    ensemble.fit(X_tr, Y_tr)
    train_predict = ensemble.predict(X_tr)
    test_predict = ensemble.predict(X_ts)
    train_error_calc = mean_squared_error(Y_tr, train_predict)
    test_error_calc = mean_squared_error(Y_ts, test_predict)

    # Results
    print("\nEnsemble Gradient Boosting Regressor")
    print("------------------------------------")
    print("Training Mean Squared Error: %.3f" % (train_error_calc))
    print("Testing Mean Squared Error: %.3f" % (test_error_calc))
    return ensemble
    
def visualize_tree(model, features, targets):
    # DOT Data
    dot_data = export_graphviz(model, feature_names=features, class_names=targets, proportion=True, filled=True, out_file=None)
    
    # Plot Tree
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('decision_tree.png')

if __name__ == '__main__':
    print("\nRegression -- Supervised Learning")
    print("-----------------------------------")
    student_data_dict = fetch_student_data()
    show_histograms(student_data_dict)
    find_correlations(student_data_dict)
    X_tr, X_ts, Y_tr, Y_ts = split_dataset(student_data_dict)
    
    decision_regressor = Decision_Tree_Regressor(X_tr, X_ts, Y_tr, Y_ts)
    most_important_features(decision_regressor, student_data_dict.keys().drop(['G3']), 5)
    
    ensemble_regressor = Ensemble(X_tr, X_ts, Y_tr, Y_ts)
    most_important_features(ensemble_regressor, student_data_dict.keys().drop(['G3']), 5)
    
    visualize_tree(decision_regressor, student_data_dict.keys().drop(['G3']), ['G3'])
