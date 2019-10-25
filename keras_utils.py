import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_sequence(group):
    group_X = np.array( group.drop(columns=['patient', 'segment_id', 'class']).values )
    group_y = np.unique(group['class'])[0]
    return group_X, group_y
  
""" Generates already scaled sequence arrays, in order to feed them to the LSTM Network"""
def sequence_generator(df, batch_size=1):
  X = df.drop(columns=['patient', 'segment_id', 'class'])
    
  scaler = StandardScaler()
  new_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  new_df['class'] = np.array(df['class'], dtype=np.int)
  new_df['patient'] = np.array(df['patient'], dtype=np.int)
  new_df['segment_id'] = np.array(df['segment_id'], dtype=np.int)
  
  sequence = new_df.groupby(['patient', 'class', 'segment_id']).apply(build_sequence)
  sequence = sequence.values
  
  while True:
    batch_X = []
    batch_y = []
    batch_counter = 0
    np.random.shuffle(sequence)
    for s in sequence:
      X, y = s
      batch_X.append(X)
      batch_y.append(y)
      batch_counter += 1
      if batch_counter == batch_size:
        yield np.array(batch_X), np.array(batch_y)
        batch_counter = 0
        batch_X = []
        batch_y = []
