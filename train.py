import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. VERİYİ YÜKLE
print("Veri yükleniyor...")
df = pd.read_csv('nba_2008-2025.csv')

# 2. VERİ TEMİZLEME VE HAZIRLIK
# Tarihi datetime formatına çevir
df['date'] = pd.to_datetime(df['date'])
# Sadece normal sezon ve playoff maçlarını al, All-Star vs eleyelim
df = df.sort_values('date')

# Hedef: Ev sahibi kazandı mı? (1: Evet, 0: Hayır)
df['target'] = (df['score_home'] > df['score_away']).astype(int)

# Amerikan Oranlarını (Moneyline) Ondalığa Çevir (Feature olarak kullanacağız)
def convert_odds(odd):
    if pd.isna(odd): return 1.0
    if odd > 0: return (odd / 100) + 1
    else: return (100 / abs(odd)) + 1

df['odds_home'] = df['moneyline_home'].apply(convert_odds)
df['odds_away'] = df['moneyline_away'].apply(convert_odds)

# 3. ÖZNİTELİK MÜHENDİSLİĞİ (BASİT MVP)
# Takımların son performanslarını (Kayan Ortalamalar) hesaplayalım
# Bu kısım biraz karmaşık olabilir, MVP için basit tutuyorum:
# "Oranlar" aslında bahis şirketlerinin tüm analizini içerdiği için çok güçlü bir feature'dır.
# Şimdilik sadece oranları ve tarih bilgisini kullanalım.

features = ['odds_home', 'odds_away']
target = 'target'

# Eksik verileri at
df = df.dropna(subset=features)

# 4. MODEL EĞİTİMİ
X = df[features]
y = df[target]

# Son %20'yi test için ayır (Zaman serisine dikkat ederek - shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Model eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. TEST VE KAYIT
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Doğruluğu: %{acc * 100:.2f}")

# Modeli kaydet (Bu dosya web sitesine yüklenecek)
joblib.dump(model, 'nba_model.pkl')
print("Model 'nba_model.pkl' olarak kaydedildi!")