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
    'LINK/USDT', 'ADA/USDT', 'NEAR/USDT', 'INJ/USDT', 'WIF/USDT', 'UNI/USDT'
]

def fetch_kucoin_data(symbol, candles=200):
    """Busca dados da KuCoin"""
    try:
        symbol = symbol.replace('/', '-')  # correção do formato
        end_time = int(time.time())
        # 3min candles => 180 segundos por candle
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

def is_price_near_ema(price: float, ema_value: float, tolerance_percent: float = 0.1) -> bool:
    if pd.isna(ema_value) or pd.isna(price):
        return False
    diff_percent = abs(price - ema_value) / ema_value * 100
    return diff_percent <= tolerance_percent

def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    try:
        if not bot_token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

def analyze_signal(ha_df, ema20, ema50):
    if len(ha_df) < 2:
        return "Sem dados suficientes", "none", False, "-", "-", "-"

    current_idx = len(ha_df) - 1
    prev_idx = current_idx - 1
    current_price = ha_df.loc[current_idx, 'ha_close']
    prev_high = ha_df.loc[prev_idx, 'ha_high']
    prev_low = ha_df.loc[prev_idx, 'ha_low']
    current_close = ha_df.loc[current_idx, 'ha_close']
    current_ema20 = ema20.iloc[current_idx]
    current_ema50 = ema50.iloc[current_idx]
    prev_ema20 = ema20.iloc[prev_idx]
    prev_ema50 = ema50.iloc[prev_idx]
    prev_price = ha_df.loc[prev_idx, 'ha_close']

    cross_label = "-"
    if not pd.isna(prev_ema20) and not pd.isna(prev_ema50) and not pd.isna(current_ema20) and not pd.isna(current_ema50):
        crossed_up = prev_ema20 <= prev_ema50 and current_ema20 > current_ema50
        crossed_down = prev_ema20 >= prev_ema50 and current_ema20 < current_ema50
        if crossed_up:
            cross_label = "Cruzou para cima (EMA20 > EMA50)"
        elif crossed_down:
            cross_label = "Cruzou para baixo (EMA20 < EMA50)"

    # Cruzamento do preço com as EMAs
    price_cross_ema20 = "-"
    price_cross_ema50 = "-"
    if not pd.isna(prev_price) and not pd.isna(current_price) and not pd.isna(prev_ema20) and not pd.isna(current_ema20):
        if prev_price <= prev_ema20 and current_price > current_ema20:
            price_cross_ema20 = "Preço cruzou acima da EMA20"
        elif prev_price >= prev_ema20 and current_price < current_ema20:
            price_cross_ema20 = "Preço cruzou abaixo da EMA20"
    if not pd.isna(prev_price) and not pd.isna(current_price) and not pd.isna(prev_ema50) and not pd.isna(current_ema50):
        if prev_price <= prev_ema50 and current_price > current_ema50:
            price_cross_ema50 = "Preço cruzou acima da EMA50"
        elif prev_price >= prev_ema50 and current_price < current_ema50:
            price_cross_ema50 = "Preço cruzou abaixo da EMA50"

    near_ema20 = is_price_near_ema(current_price, current_ema20)
    near_ema50 = is_price_near_ema(current_price, current_ema50)

    if near_ema20 and near_ema50:
        return "📍 Próximo de EMA20 e EMA50", "near-both", True, cross_label, price_cross_ema20, price_cross_ema50
    if near_ema50:
        return "📍 Próximo de EMA50", "near-ema50", True, cross_label, price_cross_ema20, price_cross_ema50
    if near_ema20:
        return "📍 Próximo de EMA20", "near-ema20", True, cross_label, price_cross_ema20, price_cross_ema50
    return "➖ Longe das EMAs", "none", False, cross_label, price_cross_ema20, price_cross_ema50

def process_token(symbol):
    df = fetch_kucoin_data(symbol)
    if df is None or len(df) < 50:
        return None

    ha_df = calculate_heikin_ashi(df)
    ha_df['ema20'] = calculate_ema(ha_df['ha_close'], 20)
    ha_df['ema50'] = calculate_ema(ha_df['ha_close'], 50)
    status, signal_type, confirmed, cross_label, price_cross_ema20, price_cross_ema50 = analyze_signal(
        ha_df, ha_df['ema20'], ha_df['ema50']
    )
    last_row = ha_df.iloc[-1]
    return {
        'token': symbol.replace('-', '/'),
        'price': f"{last_row['ha_close']:.8f}",
        'ema20': f"{last_row['ema20']:.8f}",
        'ema50': f"{last_row['ema50']:.8f}",
        'status': status,
        'signal_type': signal_type,
        'confirmed': confirmed,
        'cross': cross_label,
        'price_cross_ema20': price_cross_ema20,
        'price_cross_ema50': price_cross_ema50,
        'last_candle_time': str(last_row['time']),
        'data': ha_df
    }

# =========================
# INTERFACE KUCOIN
# =========================

def show_kucoin_page():
    st.title("📈 Analisador de Criptomoedas - KuCoin")
    st.markdown("**Estratégia:** Heikin Ashi + EMAs (3min) - **Horário de Brasília**")

    st.sidebar.header("⚙️ Configurações")
    st.sidebar.subheader("Alertas e Monitoramento")
    enable_monitor = st.sidebar.checkbox("Ativar monitoramento automático (3 min)", value=False)
    bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    chat_id = st.sidebar.text_input("Telegram Chat ID")
    send_on_near = st.sidebar.checkbox("Alertar quando preço próximo das EMAs", value=True)
    send_on_price_cross = st.sidebar.checkbox("Alertar cruzamento Preço x EMAs", value=True)
    send_on_ema_cross = st.sidebar.checkbox("Alertar cruzamento EMA20 x EMA50", value=True)

    if st.sidebar.button("🔄 Atualizar Todos os Dados", type="primary") or enable_monitor:
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

            # Envio de alertas Telegram com deduplicação por candle
            if enable_monitor and results:
                if 'last_alerts' not in st.session_state:
                    st.session_state['last_alerts'] = {}
                for item in results:
                    token_name = item['token']
                    candle_time = item.get('last_candle_time', '')
                    last_sent_for_token = st.session_state['last_alerts'].get(token_name)
                    should_send = last_sent_for_token != candle_time

                    messages = []
                    if send_on_near and item['status'].startswith("📍 Próximo"):
                        messages.append(f"{token_name}: {item['status']}")
                    if send_on_price_cross and (item.get('price_cross_ema20') not in (None, '-') or item.get('price_cross_ema50') not in (None, '-')):
                        if item.get('price_cross_ema20') and item['price_cross_ema20'] != '-':
                            messages.append(f"{token_name}: {item['price_cross_ema20']}")
                        if item.get('price_cross_ema50') and item['price_cross_ema50'] != '-':
                            messages.append(f"{token_name}: {item['price_cross_ema50']}")
                    if send_on_ema_cross and item.get('cross') not in (None, '-'):
                        messages.append(f"{token_name}: {item['cross']}")

                    if should_send and messages and bot_token and chat_id:
                        text = "\n".join(messages)
                        ok = send_telegram_message(bot_token, chat_id, text)
                        if ok:
                            st.session_state['last_alerts'][token_name] = candle_time
                if bot_token and chat_id:
                    st.sidebar.success("Alertas do ciclo enviados (se houve sinais novos).")
                else:
                    st.sidebar.info("Informe Bot Token e Chat ID para enviar alertas.")

    if 'last_update' in st.session_state:
        st.sidebar.info(f"🕒 Última atualização: {st.session_state['last_update']}")

    # Auto refresh via JavaScript a cada 3 minutos quando monitoramento ativo
    if enable_monitor:
        st.markdown(
            """
            <script>
            setTimeout(function(){ window.location.reload(); }, 180000);
            </script>
            """,
            unsafe_allow_html=True,
        )

    if 'analysis_data' in st.session_state and st.session_state['analysis_data']:
        st.header("📊 Resultados da Análise")

        df_table = pd.DataFrame([{
            'Token': item['token'],
            'Preço': item['price'],
            'EMA 20': item['ema20'],
            'EMA 50': item['ema50'],
            'Preço x EMA20': item.get('price_cross_ema20', '-'),
            'Preço x EMA50': item.get('price_cross_ema50', '-'),
            'Cruzamento': item.get('cross', '-'),
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
