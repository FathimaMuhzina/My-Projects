import pandas as pd
import pickle

df=pd.read_csv('ready_to_model.csv')
df=df.drop(['Unnamed: 0'], axis=1)
#Separating X and y
X=df.drop(['Used'], axis=1)
y=df['Used']

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

dt=DecisionTreeRegressor(random_state=1)
dt.fit(X,y)
bag = BaggingRegressor(dt)
bag.fit(X,y)


pickle.dump(bag,open('model.pkl','wb'))
