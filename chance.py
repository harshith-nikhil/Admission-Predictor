import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

# read the dataset
grad_df = pd.read_csv('Admission_Predict.csv')
# read the test dataset
test = pd.read_csv('Admission_Predict_Ver1.1.csv')
# train the model
regr = linear_model.LinearRegression()
x = np.asanyarray(grad_df[['GRE Score', 'TOEFL Score', 'CGPA', 'University Rating', 'SOP', 'LOR ']])
y = np.asanyarray(grad_df[['Chance of Admit ']])
regr.fit(x, y)
# pickling
pickle.dump(regr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
