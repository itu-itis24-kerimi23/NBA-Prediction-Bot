import streamlit as st
import joblib
import pandas as pd

# Sayfa Ayarları
st.set_page_config(page_title="NBA Tahmincisi", page_icon="🏀")

# Başlık
st.title("🏀 NBA Maç Tahmin Botu")
st.write("Makine Öğrenimi modeli ile maç sonucunu tahmin et.")

# 1. Modeli Yükle
try:
    model = joblib.load('nba_model.pkl')
except:
    st.error("Model dosyası bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")
    st.stop()

# 2. Kullanıcı Girişi (Kenar Çubuğu)
st.sidebar.header("Maç Verileri")
team_home = st.sidebar.text_input("Ev Sahibi Takım", "Lakers")
team_away = st.sidebar.text_input("Deplasman Takımı", "Celtics")

st.sidebar.subheader("Bahis Oranları (Moneyline)")
st.sidebar.info("Örnek: -150 (Favori) veya +130 (Underdog)")
ml_home = st.sidebar.number_input("Ev Sahibi Oranı", value=-150)
ml_away = st.sidebar.number_input("Deplasman Oranı", value=130)

# Oran Dönüştürücü Fonksiyon (Aynısı)
def convert_odds(odd):
    if odd > 0: return (odd / 100) + 1
    else: return (100 / abs(odd)) + 1

# 3. Tahmin Butonu
if st.button("MAÇI TAHMİN ET"):
    # Girdileri hazırla
    decimal_home = convert_odds(ml_home)
    decimal_away = convert_odds(ml_away)
    
    input_data = pd.DataFrame({
        'odds_home': [decimal_home],
        'odds_away': [decimal_away]
    })
    
    # Tahmin Yap
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Sonucu Göster
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ev Sahibi")
        st.write(f"**{team_home}**")
        st.metric(label="Kazanma İhtimali", value=f"%{probability[1]*100:.1f}")
        
    with col2:
        st.subheader("Deplasman")
        st.write(f"**{team_away}**")
        st.metric(label="Kazanma İhtimali", value=f"%{probability[0]*100:.1f}")
    
    st.divider()
    
    if prediction == 1:
        st.success(f"🏆 Tahmin: **{team_home}** Kazanır!")
    else:
        st.error(f"🏆 Tahmin: **{team_away}** Kazanır!")
        
    # Value Bet Analizi (Basit)
    implied_prob_home = 1 / decimal_home
    my_prob_home = probability[1]
    
    st.subheader("💡 Bahis Analizi")
    if my_prob_home > implied_prob_home:
        st.info(f"Değerli Bahis! Model {team_home} takımına bahisten daha fazla şans veriyor. (Model: %{my_prob_home*100:.0f} vs Bahis: %{implied_prob_home*100:.0f})")
    elif probability[0] > (1/decimal_away):
        st.info(f"Değerli Bahis! Model {team_away} takımına bahisten daha fazla şans veriyor.")
    else:
        st.warning("Bu maçta riskli veya değersiz oranlar var.")