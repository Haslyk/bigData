import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Veri setini yükleme
final_dataset = pd.read_csv('../../veri_setleri/final_dataset.csv')

# Gereksiz sütunları çıkarma
final_dataset = final_dataset.drop(columns=['Unnamed: 0'])

# Eksik değerlerin işlenmesi
final_dataset_cleaned = final_dataset.dropna(axis=1, thresh=0.5 * len(final_dataset))

for column in final_dataset_cleaned.columns:
    if final_dataset_cleaned[column].dtype == 'object':
        final_dataset_cleaned[column].fillna(final_dataset_cleaned[column].mode()[0], inplace=True)
    else:
        final_dataset_cleaned[column].fillna(final_dataset_cleaned[column].mean(), inplace=True)

# Kategorik verilerin dummies ile işlenmesi
final_dataset_encoded = pd.get_dummies(final_dataset_cleaned, columns=['occupation_status'])

# Seçilen özellikler
selected_features = ['age', 'gender', 'relationship_status', 'phq9_q2', 'phq9_q7','gad7_q4', 'acha_q2', 'social_media_time']

final_dataset_encoded['occupation_status'] = (
    final_dataset_encoded['occupation_status_Retired'] * 1 +
    final_dataset_encoded['occupation_status_Salaried Worker'] * 2 +
    final_dataset_encoded['occupation_status_School Student'] * 3 +
    final_dataset_encoded['occupation_status_University Student'] * 4
)

selected_features.append('occupation_status')

# Eğitim ve test verisi olarak bölme
X = final_dataset_encoded[selected_features]
y = final_dataset_encoded['social_media_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli kaydetme
joblib.dump(model, 'social_media_addiction_model.pkl')

# Tahminler ve performans değerlendirmesi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Eğitimde kullanılan özellik adlarını kaydetme
with open('feature_names.pkl', 'wb') as f:
    joblib.dump(selected_features, f)
