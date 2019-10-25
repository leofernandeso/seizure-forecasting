import numpy as np
import pandas as pd
import pickle
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def aggregate_predictions(predictions, agg_method):
    if agg_method == 'avg_proba':
        interictal_mean_prob = np.mean(predictions[:, 0])
        preictal_mean_prob = np.mean(predictions[:, 1])
        return np.argmax([interictal_mean_prob, preictal_mean_prob])
    elif agg_method == 'max_proba':
        interictal_max_prob = np.max(predictions[:, 0])
        preictal_max_prob = np.max(predictions[:, 1])
        return np.argmax([interictal_max_prob, preictal_max_prob])


def prepare_and_predict(group, model, scaler, pca, agg_method):
    X = group.drop(columns=['patient', 'segment_id', 'class'])
    y = np.unique(group['class'])

    if len(y) != 1:
        print('Error in grouping classes!')
    else:
        y = y[0]

    # standardization
    X = np.array(X)
    X = scaler.transform(X)

    # pca
    if pca:
        X = pca.transform(X)

    predictions = model.predict_proba(X)

    result = aggregate_predictions(predictions, agg_method) 
    return result, y 


# Performs cross validation given two arrays. The first array must contain a list of all
# training dataframes, while the second must contain a list with all evaluation dataframes.
# In order to perform the cross-validation, the parameter model must be a non fitted model (classifier).
def cross_validate(classifier, pca, train_dfs, validation_dfs, agg_method):
    count = 1
    for train_df, eval_df in zip(train_dfs, validation_dfs):
        # Dropping irrelevant columns and separating into X and y
        X_train = train_df.drop(columns=['patient', 'segment_id', 'class'])
        y_train = train_df['class']
        
        # Scaling
        print('Performing Cross-Validation in fold {}'.format(count))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Fitting model with subfold data and performing evaluation
        print('Fitting classifier...\n')
        classifier.fit(X_train, y_train)
        print('Evaluating fold...\n')
        evaluate_from_df(classifier, scaler, pca, eval_df, agg_method)
        
        count += 1


def print_plot_features_ranks(clf, X, figsize=(10, 15), plot_max=50, print_rank=False):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    most_important_features = []
    most_important_importances = []
    for idx in indices:
        if print_rank:
            print('Features ranking : \n')
            print(X.columns[idx])
        if len(most_important_features) <= plot_max:
            most_important_features.append(X.columns[idx])
            most_important_importances.append(importances[idx])
        
    plt.figure(figsize=figsize)
    plt.title("Feature importances - n = {}".format(plot_max))
    plt.barh(most_important_features[::-1], most_important_importances[::-1],
            color="r", align="center", height=0.5)
    plt.yticks(np.arange(0, len(most_important_features), step=1), most_important_features[::-1])
    plt.xlim([0, max(most_important_importances)*1.2])
    plt.show()


def evaluate_from_df(model, scaler, pca, validation_data, agg_method):

    segment_data = validation_data.groupby(['patient', 'class', 'segment_id']) \
                        .apply(prepare_and_predict, model=model, scaler=scaler, pca=pca, agg_method=agg_method)
    segment_data = segment_data.reset_index()
    predictions_array = []
    y_array = []
    for pred, y in np.array(segment_data[segment_data.columns[-1]]):
        predictions_array += [pred]
        y_array += [y]
    
    predictions_array = np.array(predictions_array)
    y_array = np.array(y_array)

    print(confusion_matrix(y_array, predictions_array))
    print(classification_report(y_array, predictions_array))
    print("ROC AUC Score : {}".format(roc_auc_score(y_array, predictions_array)))

def main():
    val_df = pd.read_csv("D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\selected_features_val.csv")
    val_df = val_df[val_df.columns[1:]]

        
    model = load('xgb.joblib')
    scaler = load('scaler.joblib')
    evaluate_from_df(model, scaler, None, val_df, agg_method='max_proba')

if __name__ == '__main__':
    main()

