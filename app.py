import streamlit as st
from datetime import datetime
import time
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from analysis_tab import show_analysis_page  # nova página de análise

# =========================
# FUNÇÕES DO ANALISADOR KUCOIN
# =========================

TOKENS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'AVAX/USDT',
    'LINK/USDT', 'PEPE/USDT', 'ADA/USDT', 'NEAR/USDT', 'INJ/USDT', 'WIF/USDT',
    'BONK/USDT', 'UNI/USDT'
]

def fetch_kucoin_data(symbol, candles=200):
    """Busca dados da KuCoin"""
    try:
        symbol = symbol.replace('/', '-')  # correção do formato
        end_time = int(time.time())
        start_time = end_time - (candles * 180)

        url = "https://api.kucoin.com/api/v1/market/candles"
        params = {
            'type': '3min',
            'symbol': symbol,
            'startAt': start_time,
            'endAt': end_time
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data['code'] == '200000' and data['data']:
            df = pd.DataFrame(data['data'], columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})

            df['time'] = pd.to_datetime(df['time'].astype(float), unit='s')
            df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')

            df = df.sort_values('time').reset_index(drop=True)
            return df.tail(candles)
    except Exception as e:
        st.error(f"Erro ao buscar {symbol}: {str(e)}")
    return None

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = 0.0

    for i in range(len(df)):
        if i == 0:
            ha_df.loc[i, 'ha_open'] = (df.loc[i, 'open'] + df.loc[i, 'close']) / 2
        else:
            ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2

    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    return ha_df

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_stoch_rsi(df, period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()

    return stoch_rsi_k, stoch_rsi_d

def analyze_signal(ha_df, ema20, ema50, stoch_rsi):
    if len(ha_df) < 2:
        return "Sem dados suficientes", "none", False

    current_idx = len(ha_df) - 1
    prev_idx = current_idx - 1
    current_price = ha_df.loc[current_idx, 'ha_close']
    current_rsi = stoch_rsi.iloc[current_idx]
    prev_high = ha_df.loc[prev_idx, 'ha_high']
    prev_low = ha_df.loc[prev_idx, 'ha_low']
    current_close = ha_df.loc[current_idx, 'ha_close']
    current_ema20 = ema20.iloc[current_idx]
    current_ema50 = ema50.iloc[current_idx]

    if pd.isna(current_rsi):
        return "➖ Sem Sinal", "none", False

    if current_rsi < 20:
        price_above_ema50 = current_price > current_ema50
        price_above_ema20 = current_price > current_ema20
        confirmed = current_close > prev_high
        if price_above_ema50 and confirmed:
            return "✅ Compra Forte Confirmada", "buy-strong", True
        elif price_above_ema50:
            return "⏳ Aguardando Confirmação (Compra Forte)", "buy-strong-pending", False
        elif price_above_ema20 and confirmed:
            return "✅ Compra Fraca Confirmada", "buy-weak", True
        elif price_above_ema20:
            return "⏳ Aguardando Confirmação (Compra Fraca)", "buy-weak-pending", False

    if current_rsi > 80:
        price_below_ema50 = current_price < current_ema50
        price_below_ema20 = current_price < current_ema20
        confirmed = current_close < prev_low
        if price_below_ema50 and confirmed:
            return "❌ Venda Forte Confirmada", "sell-strong", True
        elif price_below_ema50:
            return "⏳ Aguardando Confirmação (Venda Forte)", "sell-strong-pending", False
        elif price_below_ema20 and confirmed:
            return "❌ Venda Fraca Confirmada", "sell-weak", True
        elif price_below_ema20:
            return "⏳ Aguardando Confirmação (Venda Fraca)", "sell-weak-pending", False

    return "➖ Sem Sinal", "none", False

def process_token(symbol):
    df = fetch_kucoin_data(symbol)
    if df is None or len(df) < 50:
        return None

    ha_df = calculate_heikin_ashi(df)
    ha_df['ema20'] = calculate_ema(ha_df['ha_close'], 20)
    ha_df['ema50'] = calculate_ema(ha_df['ha_close'], 50)
    ha_df['stoch_rsi'], ha_df['stoch_rsi_d'] = calculate_stoch_rsi(ha_df)

    status, signal_type, confirmed = analyze_signal(
        ha_df, ha_df['ema20'], ha_df['ema50'], ha_df['stoch_rsi']
    )
    last_row = ha_df.iloc[-1]
    return {
        'token': symbol.replace('-', '/'),
        'price': f"{last_row['ha_close']:.8f}",
        'rsi': f"{last_row['stoch_rsi']:.2f}" if not pd.isna(last_row['stoch_rsi']) else "N/A",
        'ema20': f"{last_row['ema20']:.8f}",
        'ema50': f"{last_row['ema50']:.8f}",
        'status': status,
        'signal_type': signal_type,
        'confirmed': confirmed,
        'data': ha_df
    }

# =========================
# INTERFACE KUCOIN
# =========================

def show_kucoin_page():
    st.title("📈 Analisador de Criptomoedas - KuCoin")
    st.markdown("**Estratégia:** Heikin Ashi + RSI Estocástico + EMAs (3min) - **Horário de Brasília**")

    st.sidebar.header("⚙️ Configurações")

    if st.sidebar.button("🔄 Atualizar Todos os Dados", type="primary"):
        with st.spinner("Buscando dados dos tokens... Isso pode levar alguns minutos..."):
            results = []
            progress_bar = st.progress(0)
            for idx, token in enumerate(TOKENS):
                token_data = process_token(token)
                if token_data:
                    results.append(token_data)
                progress_bar.progress((idx + 1) / len(TOKENS))
                time.sleep(0.5)

            results.sort(key=lambda x: x['signal_type'])
            st.session_state['analysis_data'] = results
            st.session_state['last_update'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            st.success(f"✅ Análise concluída! {len(results)} tokens processados.")

    if 'last_update' in st.session_state:
        st.sidebar.info(f"🕒 Última atualização: {st.session_state['last_update']}")

    if 'analysis_data' in st.session_state and st.session_state['analysis_data']:
        st.header("📊 Resultados da Análise")

        df_table = pd.DataFrame([{
            'Token': item['token'],
            'Preço': item['price'],
            'RSI': item['rsi'],
            'EMA 20': item['ema20'],
            'EMA 50': item['ema50'],
            'Status': item['status']
        } for item in st.session_state['analysis_data']])

        st.dataframe(df_table, use_container_width=True, height=600)
    else:
        st.info("👆 Clique em 'Atualizar Todos os Dados' na barra lateral para começar a análise!")

# =========================
# MENU PRINCIPAL
# =========================

def main():
    st.sidebar.title("🧭 Navegação")
    page = st.sidebar.radio("Escolha uma página:", ["Analisador KuCoin", "Análise de Resultados"])

    if page == "Analisador KuCoin":
        show_kucoin_page()
    else:
        show_analysis_page()

if __name__ == "__main__":
    st.set_page_config(page_title="Analisador Cripto KuCoin", page_icon="📈", layout="wide")
    main()
