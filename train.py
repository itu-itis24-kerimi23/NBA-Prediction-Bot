import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. VERİYİ YÜKLE
print("Veri yükleniyor...")
df = pd.read_csv('nba_2008-2025.csv')

# 2. VERİ HAZIRLIĞI
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Hedef: Ev sahibi kazandı mı?
df['target'] = (df['score_home'] > df['score_away']).astype(int)

# Amerikan Oranlarını Eğitirken Decimal'e Çeviriyoruz
def convert_odds(odd):
    if pd.isna(odd): return 1.0
    if odd > 0: return (odd / 100) + 1
    else: return (100 / abs(odd)) + 1

df['odds_home'] = df['moneyline_home'].apply(convert_odds)
df['odds_away'] = df['moneyline_away'].apply(convert_odds)

# Takım Listesini Al (Alfabetik Sırayla)
teams = sorted(df['home'].unique().tolist())
print(f"Toplam {len(teams)} takım bulundu.")

features = ['odds_home', 'odds_away']
df = df.dropna(subset=features)

# 3. MODEL EĞİTİMİ
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Model eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test Sonucu
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Doğruluğu: %{acc * 100:.2f}")

# 4. KAYIT (Modeli VE Takım Listesini paketliyoruz)
# Sadece modeli değil, bir sözlük (dictionary) kaydediyoruz
model_data = {
    'model': model,
    'teams': teams
}

joblib.dump(model_data, 'nba_model.pkl')
print("Model ve takım listesi 'nba_model.pkl' içine kaydedildi!")
