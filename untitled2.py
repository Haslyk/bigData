import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Seçilen özellikler ve sosyal medya kullanım süresi
selected_features = ['age', 'gender', 'relationship_status', 'phq9_q2', 'phq9_q7', 'gad7_q1', 'gad7_q4', 'acha_q2', 'social_media_time']

# Grafikler için koyu renk paleti ayarlama
sns.set_palette("dark")

# Uyku Sorunları Sıklığı Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='phq9_q2', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Uyku Sorunları Sıklığı Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='Uyku Sorunları Sıklığı', labels=['Hiçbir Zaman', 'Nadiren', 'Bazen', 'Sık Sık', 'Her Zaman'])
plt.show()


# Depresyon Sıklığı Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='gad7_q1', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Depresyon Sıklığı Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='Depresyon Sıklığı', labels=['Hiçbir Zaman', 'Nadiren', 'Bazen', 'Sık Sık', 'Her Zaman'])
plt.show()

# Dikkat Dağınıklığı Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='gad7_q4', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Dikkat Dağınıklığı Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='Dikkat Dağınıklığı', labels=['Hiçbir Zaman', 'Nadiren', 'Bazen', 'Sık Sık', 'Her Zaman'])
plt.show()

# Duygu Durumları Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='acha_q2', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Duygu Durumları Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='Duygu Durumları', labels=['Mutluluk', 'Öfke', 'Nötr', 'Kaygı', 'Can Sıkıntısı', 'Üzüntü'])
plt.show()

# Endişe Duyma Sıklığı Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='phq9_q7', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Endişe Duyma Sıklığı Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='Endişe Duyma Sıklığı', labels=['Hiçbir Zaman', 'Nadiren', 'Bazen', 'Sık Sık', 'Her Zaman'])
plt.show()

# Günlük Aktivitelerde İlgi Değişimi Sıklığı Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='phq9_q2', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Günlük Aktivitelerde İlgi Değişimi Sıklığı Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='İlgi Değişimi Sıklığı', labels=['Hiçbir Zaman', 'Nadiren', 'Bazen', 'Sık Sık', 'Her Zaman'])
plt.show()


# Sosyal Medya Kullanmayınca Huzursuzluk Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(final_dataset_encoded, x='social_media_time', hue='phq9_q7', multiple='stack', kde=False, palette="dark")
plt.title("Sosyal Medya Kullanım Süresi ile Sosyal Medya Kullanmayınca Huzursuzluk Dağılımı")
plt.xlabel("Sosyal Medya Kullanım Süresi (saat)")
plt.ylabel("Kişi Sayısı")
plt.legend(title='Huzursuzluk Sıklığı', labels=['Hiçbir Zaman', 'Nadiren', 'Bazen', 'Sık Sık', 'Her Zaman'])
plt.show()
