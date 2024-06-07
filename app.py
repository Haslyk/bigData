from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Modeli ve özellik adlarını yükleme
model = joblib.load('social_media_addiction_model.pkl')
selected_features = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan gelen verileri alma
        data = request.get_json()

        # Boş stringleri 0'a veya uygun default değerlere çevirme
        def convert_to_float(value):
            if value.strip('"') == '':
                return 0.0
            return float(value.strip('"'))

        age = convert_to_float(data['age'])
        gender = convert_to_float(data['gender'])
        relationship_status = convert_to_float(data['relationship_status'])
        social_media_time = convert_to_float(data['social_media_time'])
        phq9_q2 = convert_to_float(data['phq9_q2'])
        phq9_q7 = convert_to_float(data['phq9_q7'])
        gad7_q4 = convert_to_float(data['gad7_q4'])
        acha_q2 = convert_to_float(data['acha_q2'])
        occupation_status = convert_to_float(data['occupation_status'])

        occupation_status_dict = {
            1: [1, 0, 0, 0],
            2: [0, 1, 0, 0],
            3: [0, 0, 1, 0],
            4: [0, 0, 0, 1]
        }

        # Mental sağlık skoru hesaplama
        mental_health_score = (
            phq9_q2 + phq9_q7 +
            gad7_q4
        )

        # Veriyi DataFrame'e dönüştürme
        new_user_data = {
            'age': [age],
            'gender': [gender],
            'relationship_status': [relationship_status],
            'social_media_time': [social_media_time],
            'phq9_q2': [phq9_q2],
            'phq9_q7': [phq9_q7],
            'gad7_q4': [gad7_q4],
            'acha_q2': [acha_q2],
            'occupation_status_Retired': [occupation_status_dict[occupation_status][0]],
            'occupation_status_Salaried Worker': [occupation_status_dict[occupation_status][1]],
            'occupation_status_School Student': [occupation_status_dict[occupation_status][2]],
            'occupation_status_University Student': [occupation_status_dict[occupation_status][3]]
        }

        new_user_df = pd.DataFrame(new_user_data)
        new_user_df = new_user_df[selected_features]

        # Tahmin yapma
        predicted_addiction_score = model.predict(new_user_df)[0]

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

        recommendation = get_recommendations(predicted_addiction_score)

        return jsonify(score=predicted_addiction_score, mental_health_score=mental_health_score, recommendation=recommendation)
    
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
