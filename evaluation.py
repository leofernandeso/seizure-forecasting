import numpy as np
import pandas as pd
import pickle
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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



val_df = pd.read_csv("D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\selected_features_val.csv")
val_df = val_df[val_df.columns[1:]]

    
model = load('xgb.joblib')
scaler = load('scaler.joblib')
evaluate_from_df(model, scaler, None, val_df, agg_method='max_proba')