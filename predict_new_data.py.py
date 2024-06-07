import joblib
import pandas as pd

# Modeli ve özellik adlarını yükleme
model = joblib.load('social_media_addiction_model.pkl')
selected_features = joblib.load('feature_names.pkl')

# Kullanıcıdan gelen veriler
new_user_data = {
    'age': [21],
    'gender': [1], 
    'relationship_status': [1], 
    'phq9_q2': [1.0], 
    'phq9_q7': [1.0],
    'gad7_q4': [1.0],
    'acha_q2': [1.0],
    'social_media_time': [2.0],
    'occupation_status_Retired': [0],
    'occupation_status_Salaried Worker': [0],
    'occupation_status_School Student': [0],
    'occupation_status_University Student': [1]
}

# Veriyi DataFrame'e dönüştürme ve özellikleri sıralama
new_user_df = pd.DataFrame(new_user_data)
new_user_df = new_user_df[selected_features]

# Tahmin yapma
predicted_addiction_score = model.predict(new_user_df)

print(f"Predicted Mental Health Score: {predicted_addiction_score[0]}")

# Öneri sistemi
def get_recommendations(score):
    if score < 5:
        return "Mental sağlığınız iyi görünüyor. Devam edin!"
    elif score < 10:
        return "Mental sağlığınızı korumak için biraz daha dikkatli olun. Rahatlamak için günlük aktiviteler ve hobilerle zaman geçirin."
    elif score < 15:
        return "Mental sağlığınızı iyileştirmek için daha fazla çaba gösterin. Sosyal medya kullanımınızı azaltmayı ve daha fazla fiziksel aktivite yapmayı düşünün."
    else:
        return "Mental sağlığınız risk altında. Bir uzmandan yardım almayı düşünebilirsiniz."

# Tahmin edilen skor üzerinden öneri sunma
recommendation = get_recommendations(predicted_addiction_score[0])
print(f"Recommendation: {recommendation}")
