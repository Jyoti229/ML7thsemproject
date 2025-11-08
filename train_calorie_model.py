# train_calorie_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Synthetic dataset for calorie prediction.
# You should expand this with real data later.
data = {
    'age': [18,22,25,30,35,40,45,50,28,32,27,60,55,48,20,26,33,38,42,29],
    'weight': [50,55,60,65,70,75,80,85,58,68,62,90,82,77,52,59,71,73,79,66],
    'height': [155,160,165,170,172,175,178,180,162,169,167,185,183,176,158,161,174,171,179,168],
    'activity_level': [1.2,1.375,1.55,1.375,1.725,1.55,1.2,1.9,1.55,1.375,1.55,1.2,1.375,1.55,1.2,1.55,1.725,1.375,1.55,1.5],
    # goal: 0=lose,1=maintain,2=gain
    'goal': [0,1,1,2,0,1,0,2,1,1,0,2,1,1,0,2,1,0,2,1],
    # These calories are synthetic — they loosely follow TDEE patterns. Replace with real labels later.
    'calories': [1500,1800,2000,2400,1700,2200,1600,3000,2100,1950,1650,3200,2600,2300,1550,2500,2250,1800,2800,2050]
}

df = pd.DataFrame(data)

X = df[['age', 'weight', 'height', 'activity_level', 'goal']]
y = df['calories']

model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'calorie_model.pkl')
print("✅ Trained and saved calorie_model.pkl")
