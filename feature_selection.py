import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

base_path = "D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem"
most_correlated_path = os.path.join(base_path, "selected_features_train.csv")
train_csv_path = "D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\extracted_features_train.csv"


def get_most_correlated(df, patient=None, correlation_thresh=0.05):

    if patient:
        df = df.loc[df['patient'] == patient]
        most_correlated_path = os.path.join(base_path, "selected_features_Pat{}Train.csv".format(patient))

    patient_col = df['patient']
    segment_id_col = df['segment_id']

    print(df.columns)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
    corr = scaled_df.corr()
    corr.to_csv(os.path.join(base_path, 'correlation.csv'))
    corr_target = abs(corr['class'])
    corr_df = df[df.columns[corr_target >= correlation_thresh]]
    corr_df['patient'] = patient_col
    corr_df['segment_id'] = segment_id_col
    corr_df.to_csv(most_correlated_path)
    print(corr_df.columns)

def main():
    df_train = pd.read_csv(train_csv_path)
    get_most_correlated(df_train, patient=2, correlation_thresh=0.04)


if __name__ == '__main__':
    main()