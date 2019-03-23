import os
import pandas as pd
from sklearn.model_selection import train_test_split


gw_data = pd.DataFrame({'filename': os.listdir('./great white shark'), 'class': 0})
hh_data = pd.DataFrame({'filename': os.listdir('./hammerhead shark'), 'class': 1})

data = pd.concat([gw_data, hh_data], ignore_index=True)
X = data['filename']
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5, shuffle=True)
