import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuração da página
st.set_page_config(
    page_title="Analisador Cripto KuCoin",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lista de tokens
TOKENS = [
    'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'HYPE-USDT', 'PUMP-USDT', 'ENA-USDT',
    'FARTCOIN-USDT', 'BONK-USDT', 'BNB-USDT', 'ADA-USDT', 'XRP-USDT', 'DOGE-USDT',
    'TRX-USDT', 'LINK-USDT', 'LTC-USDT', 'PENGU-USDT', 'DOT-USDT', 'BCH-USDT',
    'SHIB-USDT', 'AVAX-USDT', 'OP-USDT', 'UNI-USDT', 'ATOM-USDT', 'ETC-USDT',
    'XLM-USDT', 'FIL-USDT', 'APT-USDT', 'SUI-USDT', 'HBAR-USDT', 'ZORA-USDT',
    'AR-USDT', 'INJ-USDT', 'PEPE-USDT', 'NEAR-USDT', 'STX-USDT', 'ALGO-USDT',
    'IMX-USDT', 'WIF-USDT', 'MINA-USDT', 'DYDX-USDT', 'TIA-USDT', 'JTO-USDT',
    'AAVE-USDT', 'PYTH-USDT', 'SAND-USDT', 'CAKE-USDT', 'BLUR-USDT', 'GMX-USDT',
    'LDO-USDT', 'FET-USDT', 'DYM-USDT', 'GMT-USDT', 'MEME-USDT', 'BOME-USDT',
    'YGG-USDT', 'RUNE-USDT', 'CELO-USDT', 'WLD-USDT', 'ONDO-USDT', 'SEI-USDT',
    'JUP-USDT', 'POPCAT-USDT', 'TAO-USDT', 'TON-USDT'
]

# Funções de cálculo
def fetch_kucoin_data(symbol, candles=200):
    """Busca dados da KuCoin"""
    try:
        end_time = int(time.time())
        start_time = end_time - (candles * 180)  # 3min = 180 segundos
        
        url = f"https://api.kucoin.com/api/v1/market/candles"
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
            df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
            df = df.sort_values('time').reset_index(drop=True)
            return df.tail(candles)
    except Exception as e:
        st.error(f"Erro ao buscar {symbol}: {str(e)}")
    return None

def calculate_heikin_ashi(df):
    """Calcula velas Heikin Ashi"""
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
    """Calcula EMA"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_stoch_rsi(df, period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    """Calcula RSI Estocástico"""
    # Calcular RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calcular Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    
    # Suavizar
    stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
    
    return stoch_rsi_k, stoch_rsi_d

def analyze_signal(ha_df, ema20, ema50, stoch_rsi):
    """Analisa sinais de compra/venda"""
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
    
    # Verificar se RSI é válido
    if pd.isna(current_rsi):
        return "➖ Sem Sinal", "none", False
    
    # SINAIS DE COMPRA
    if current_rsi < 20:
        price_above_ema50 = current_price > current_ema50
        price_above_ema20 = current_price > current_ema20
        confirmed = current_close > prev_high
        
        if price_above_ema50:
            if confirmed:
                return "✅ Compra Forte Confirmada", "buy-strong", True
            else:
                return "⏳ Aguardando Confirmação (Compra Forte)", "buy-strong-pending", False
        elif price_above_ema20:
            if confirmed:
                return "✅ Compra Fraca Confirmada", "buy-weak", True
            else:
                return "⏳ Aguardando Confirmação (Compra Fraca)", "buy-weak-pending", False
    
    # SINAIS DE VENDA
    if current_rsi > 80:
        price_below_ema50 = current_price < current_ema50
        price_below_ema20 = current_price < current_ema20
        confirmed = current_close < prev_low
        
        if price_below_ema50:
            if confirmed:
                return "❌ Venda Forte Confirmada", "sell-strong", True
            else:
                return "⏳ Aguardando Confirmação (Venda Forte)", "sell-strong-pending", False
        elif price_below_ema20:
            if confirmed:
                return "❌ Venda Fraca Confirmada", "sell-weak", True
            else:
                return "⏳ Aguardando Confirmação (Venda Fraca)", "sell-weak-pending", False
    
    return "➖ Sem Sinal", "none", False

def process_token(symbol):
    """Processa um token completo"""
    df = fetch_kucoin_data(symbol)
    
    if df is None or len(df) < 50:
        return None
    
    ha_df = calculate_heikin_ashi(df)
    
    # Calcular indicadores
    ha_df['ema20'] = calculate_ema(ha_df['ha_close'], 20)
    ha_df['ema50'] = calculate_ema(ha_df['ha_close'], 50)
    ha_df['stoch_rsi'], ha_df['stoch_rsi_d'] = calculate_stoch_rsi(ha_df)
    
    # Analisar sinal
    status, signal_type, confirmed = analyze_signal(
        ha_df, 
        ha_df['ema20'], 
        ha_df['ema50'], 
        ha_df['stoch_rsi']
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

def plot_chart(token_data):
    """Cria gráfico Plotly com Heikin Ashi"""
    df = token_data['data']
    
    fig = go.Figure()
    
    # Velas Heikin Ashi
    colors = ['green' if row['ha_close'] >= row['ha_open'] else 'red' 
              for idx, row in df.iterrows()]
    
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['ha_open'],
        high=df['ha_high'],
        low=df['ha_low'],
        close=df['ha_close'],
        name='Heikin Ashi',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # EMAs
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['ema20'],
        name='EMA 20',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['ema50'],
        name='EMA 50',
        line=dict(color='orange', width=2)
    ))
    
    # Marcar sinais confirmados
    buy_signals = df[
        (df['stoch_rsi'] < 20) & 
        (df['ha_close'] > df['ha_high'].shift(1))
    ]
    
    sell_signals = df[
        (df['stoch_rsi'] > 80) & 
        (df['ha_close'] < df['ha_low'].shift(1))
    ]
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['time'],
            y=buy_signals['ha_low'],
            mode='markers',
            name='Sinal Compra',
            marker=dict(color='lime', size=12, symbol='triangle-up')
        ))
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['time'],
            y=sell_signals['ha_high'],
            mode='markers',
            name='Sinal Venda',
            marker=dict(color='red', size=12, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=f"{token_data['token']} - Heikin Ashi (3min)",
        xaxis_title="Data/Hora",
        yaxis_title="Preço (USDT)",
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

def plot_rsi(token_data):
    """Cria gráfico do RSI Estocástico"""
    df = token_data['data']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['stoch_rsi'],
        name='Stoch RSI',
        line=dict(color='cyan', width=2)
    ))
    
    # Linhas de referência
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Sobrecomprado (80)")
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Sobrevendido (20)")
    
    fig.update_layout(
        title="RSI Estocástico",
        xaxis_title="Data/Hora",
        yaxis_title="RSI (%)",
        height=300,
        template='plotly_dark',
        yaxis_range=[0, 100]
    )
    
    return fig

# Interface Streamlit
def main():
    st.title("📈 Analisador de Criptomoedas - KuCoin")
    st.markdown("**Estratégia:** Heikin Ashi + RSI Estocástico + EMAs (3min)")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    if st.sidebar.button("🔄 Atualizar Todos os Dados", type="primary"):
        with st.spinner("Buscando dados de 64 tokens... Isso pode levar alguns minutos..."):
            results = []
            progress_bar = st.progress(0)
            
            for idx, token in enumerate(TOKENS):
                token_data = process_token(token)
                if token_data:
                    results.append(token_data)
                progress_bar.progress((idx + 1) / len(TOKENS))
                time.sleep(0.5)  # Evitar rate limit
            
            # Ordenar resultados
            def get_sort_key(item):
                order = {
                    'buy-strong': 1, 'buy-weak': 2, 
                    'sell-strong': 3, 'sell-weak': 4,
                    'buy-strong-pending': 5, 'buy-weak-pending': 6,
                    'sell-strong-pending': 7, 'sell-weak-pending': 8,
                    'none': 9
                }
                confirmed_bonus = 0 if item['confirmed'] else 10
                return order.get(item['signal_type'], 99) + confirmed_bonus
            
            results.sort(key=get_sort_key)
            
            st.session_state['analysis_data'] = results
            st.session_state['last_update'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            st.success(f"✅ Análise concluída! {len(results)} tokens processados.")
    
    if 'last_update' in st.session_state:
        st.sidebar.info(f"🕒 Última atualização: {st.session_state['last_update']}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Legenda de Sinais")
    st.sidebar.markdown("""
    - 🟢 **Compra Forte**: RSI<20 + Preço>EMA50
    - 🟡 **Compra Fraca**: RSI<20 + Preço>EMA20
    - 🔴 **Venda Forte**: RSI>80 + Preço<EMA50
    - 🟠 **Venda Fraca**: RSI>80 + Preço<EMA20
    - ⏳ **Aguardando**: Precisa confirmar próxima vela
    """)
    
    # Mostrar tabela de resultados
    if 'analysis_data' in st.session_state and st.session_state['analysis_data']:
        st.header("📊 Resultados da Análise")
        
        # Criar DataFrame para tabela
        table_data = []
        for item in st.session_state['analysis_data']:
            table_data.append({
                'Token': item['token'],
                'Preço': item['price'],
                'RSI': item['rsi'],
                'EMA 20': item['ema20'],
                'EMA 50': item['ema50'],
                'Status': item['status']
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Aplicar cores
        def highlight_status(val):
            if '✅' in val and 'Forte' in val:
                return 'background-color: #22c55e; color: white; font-weight: bold'
            elif '✅' in val:
                return 'background-color: #86efac; color: black; font-weight: bold'
            elif '❌' in val and 'Forte' in val:
                return 'background-color: #ef4444; color: white; font-weight: bold'
            elif '❌' in val:
                return 'background-color: #fca5a5; color: black; font-weight: bold'
            elif '⏳' in val:
                return 'background-color: #eab308; color: black; font-weight: bold'
            return ''
        
        styled_df = df_table.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Seleção de token para visualização
        st.header("📈 Visualização Detalhada")
        
        token_names = [item['token'] for item in st.session_state['analysis_data']]
        selected_token = st.selectbox("Selecione um token para ver o gráfico:", token_names)
        
        if selected_token:
            token_data = next(item for item in st.session_state['analysis_data'] if item['token'] == selected_token)
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Preço Atual", f"${token_data['price']}")
            with col2:
                st.metric("RSI Estocástico", token_data['rsi'])
            with col3:
                st.metric("EMA 20", f"${token_data['ema20']}")
            with col4:
                st.metric("EMA 50", f"${token_data['ema50']}")
            
            # Status
            st.info(f"**Status:** {token_data['status']}")
            
            # Gráficos
            st.plotly_chart(plot_chart(token_data), use_container_width=True)
            st.plotly_chart(plot_rsi(token_data), use_container_width=True)
    
    else:
        st.info("👆 Clique em 'Atualizar Todos os Dados' na barra lateral para começar a análise!")

if __name__ == "__main__":
    main()
