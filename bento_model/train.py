import bentoml
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Thyroid_train.csv')
x = df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']].values
y= df[['diagnosis']].values

rfc = RandomForestClassifier()
rfc.fit(x,y)

saved_model = bentoml.sklearn.save_model("thyroid_rfc",rfc)
print(f'Model saved:{saved_model}')