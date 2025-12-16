import streamlit as st
import joblib
import pandas as pd

# Sayfa Ayarları
st.set_page_config(page_title="NBA Tahmincisi", page_icon="🏀")

st.title("🏀 NBA Maç Tahmin Botu")

# 1. Modeli ve Takım Listesini Yükle
try:
    data = joblib.load('nba_model.pkl')
    # Paketin içinden modeli ve listeyi çıkarıyoruz
    model = data['model']
    teams = data['teams']
except:
    st.error("Model dosyası eksik veya hatalı! Lütfen train.py dosyasını tekrar çalıştırın.")
    st.stop()

# 2. Kullanıcı Girişi (Kenar Çubuğu)
st.sidebar.header("Maç Seçimi")

# Takımları Listeden Seçtirme (Selectbox)
team_home = st.sidebar.selectbox("Ev Sahibi Takım", teams, index=0) # İlk sıradaki seçili gelir
team_away = st.sidebar.selectbox("Deplasman Takımı", teams, index=1) # İkinci sıradaki seçili gelir

st.sidebar.divider()

st.sidebar.header("Bahis Oranları (Decimal)")
st.sidebar.info("Örnek: 1.66, 2.40 gibi ondalık oran giriniz.")

# Ondalık Giriş (Step 0.01 sayesinde 1.66 gibi girilebilir)
odds_home = st.sidebar.number_input("Ev Sahibi Oranı (1.xx)", min_value=1.01, value=1.50, step=0.01, format="%.2f")
odds_away = st.sidebar.number_input("Deplasman Oranı (1.xx)", min_value=1.01, value=2.50, step=0.01, format="%.2f")

# 3. Tahmin Butonu
if st.button("MAÇI TAHMİN ET"):
    # Girdileri hazırla (Artık çeviri yapmıyoruz, direkt giriyoruz)
    input_data = pd.DataFrame({
        'odds_home': [odds_home],
        'odds_away': [odds_away]
    })
    
    # Tahmin Yap
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    prob_home = probability[1]
    prob_away = probability[0]

    # Sonucu Göster
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {team_home}")
        st.metric(label="Kazanma İhtimali", value=f"%{prob_home*100:.1f}")
        
    with col2:
        st.subheader(f"✈️ {team_away}")
        st.metric(label="Kazanma İhtimali", value=f"%{prob_away*100:.1f}")
    
    st.divider()
    
    if prediction == 1:
        st.success(f"🏆 Tahmin: **{team_home}** Kazanır!")
    else:
        st.error(f"🏆 Tahmin: **{team_away}** Kazanır!")
        
    # Value Bet Analizi
    # Bahis şirketinin olasılığı = 1 / Oran
    implied_prob_home = 1 / odds_home
    
    st.subheader("💡 Bahis Analizi")
    
    # Modelin tahmini > Bahis şirketinin tahmini ise Value vardır
    if prob_home > implied_prob_home:
        roi = (prob_home * odds_home) - 1
        st.info(f"✅ **Değerli Bahis (Value Bet)!**\n\nModel, {team_home} takımına bahis şirketinden daha fazla güveniyor.\n(Beklenen Kâr: %{roi*100:.1f})")
    elif prob_away > (1 / odds_away):
        roi = (prob_away * odds_away) - 1
        st.info(f"✅ **Değerli Bahis (Value Bet)!**\n\nModel, {team_away} takımına bahis şirketinden daha fazla güveniyor.\n(Beklenen Kâr: %{roi*100:.1f})")
    else:
        st.warning("⚠️ **Pas Geç.** Oranlar riske girmeye değecek kadar yüksek değil.")
