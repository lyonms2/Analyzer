import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from typing import Dict, Tuple, List
import time
from ema_barriers_config import EMA_BARRIERS, CRYPTO_LIST

# Configuração da página
st.set_page_config(
    page_title="Analisador de Criptomoedas - Barreiras EMA",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KuCoinCryptoAnalyzer:
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.crypto_symbols = CRYPTO_LIST
        
    def get_kucoin_data(self, symbol: str, timeframe: str = "1day", days: int = 200) -> pd.DataFrame:
        """Busca dados da KuCoin API"""
        try:
            # Converter timeframe para KuCoin format
            kucoin_timeframes = {
                "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1hour", "4h": "4hour", "1d": "1day", "1w": "1week"
            }
            
            kucoin_tf = kucoin_timeframes.get(timeframe, "1day")
            
            # Calcular timestamps
            end_time = int(time.time())
            start_time = end_time - (days * 24 * 60 * 60)
            
            # Fazer requisição para KuCoin
            url = f"{self.base_url}/api/v1/market/candles"
            params = {
                'symbol': symbol.replace('/', '-'),
                'type': kucoin_tf,
                'startAt': start_time,
                'endAt': end_time
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] != '200000' or not data['data']:
                st.warning(f"Dados não encontrados para {symbol}")
                return pd.DataFrame()
            
            # Converter para DataFrame
            df = pd.DataFrame(data['data'], columns=[
                'timestamp', 'Open', 'Close', 'High', 'Low', 'Volume', 'Turnover'
            ])
            
            # Processar dados
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Converter para float
            numeric_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            st.error(f"Erro ao buscar dados da KuCoin para {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_ema_barriers(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calcula as barreiras EMA específicas para cada ativo"""
        df = data.copy()
        
        if symbol in EMA_BARRIERS:
            barriers = EMA_BARRIERS[symbol]
            df['Barreira_1'] = ta.trend.EMAIndicator(
                close=df['Close'], window=barriers['barreira_1']
            ).ema_indicator()
            df['Barreira_2'] = ta.trend.EMAIndicator(
                close=df['Close'], window=barriers['barreira_2']
            ).ema_indicator()
            df['Barreira_3'] = ta.trend.EMAIndicator(
                close=df['Close'], window=barriers['barreira_3']
            ).ema_indicator()
        else:
            # Valores padrão se não configurado
            df['Barreira_1'] = ta.trend.EMAIndicator(close=df['Close'], window=21).ema_indicator()
            df['Barreira_2'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
            df['Barreira_3'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()
            
        return df
    
    def calculate_rsi_75(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula RSI de 75 períodos"""
        df = data.copy()
        df['RSI_75'] = ta.momentum.RSIIndicator(
            close=df['Close'], window=75
        ).rsi()
        return df
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calcula Estocástico padrão"""
        df = data.copy()
        
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'], 
            close=df['Close'],
            window=k_period,
            smooth_window=d_period
        )
        
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        return df
    
    def detect_barrier_respect(self, data: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
        """Detecta quando o preço toca e respeita uma barreira"""
        df = data.copy()
        
        # Inicializar colunas de respeito às barreiras
        df['Respect_B1'] = 0
        df['Respect_B2'] = 0  
        df['Respect_B3'] = 0
        
        for i in range(lookback, len(df)):
            current_close = df['Close'].iloc[i]
            previous_closes = df['Close'].iloc[i-lookback:i]
            
            # Verificar cada barreira
            barriers = ['Barreira_1', 'Barreira_2', 'Barreira_3']
            respect_cols = ['Respect_B1', 'Respect_B2', 'Respect_B3']
            
            for barrier, respect_col in zip(barriers, respect_cols):
                barrier_value = df[barrier].iloc[i]
                
                if pd.isna(barrier_value):
                    continue
                
                # Verificar se houve toque e respeito
                # Toque: preço chegou próximo da barreira (±0.5%)
                touch_threshold = barrier_value * 0.005
                
                # Verificar se algum dos preços recentes tocou a barreira
                touched = False
                for prev_close in previous_closes:
                    if abs(prev_close - barrier_value) <= touch_threshold:
                        touched = True
                        break
                
                if touched:
                    # Verificar se respeitou (voltou na direção da tendência)
                    if current_close > df['Barreira_3'].iloc[i]:  # Tendência de alta
                        if current_close > barrier_value:  # Respeitou e voltou para cima
                            df.loc[df.index[i], respect_col] = 1
                    else:  # Tendência de baixa
                        if current_close < barrier_value:  # Respeitou e voltou para baixo
                            df.loc[df.index[i], respect_col] = -1
        
        return df
    
    def apply_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica a nova estratégia com barreiras"""
        df = data.copy()
        
        # Determinar tendência geral (Barreira 3)
        df['Trend'] = np.where(df['Close'] > df['Barreira_3'], 1, -1)  # 1=Bull, -1=Bear
        
        # Sinais RSI (75 períodos)
        df['RSI_Bull'] = df['RSI_75'] > 50
        df['RSI_Bear'] = df['RSI_75'] < 50
        
        # Sinais Estocástico
        df['Stoch_Oversold'] = df['Stoch_K'] < 20  # Sinal de VENDA
        df['Stoch_Overbought'] = df['Stoch_K'] > 80  # Sinal de COMPRA
        
        # Detectar respeito às barreiras
        df = self.detect_barrier_respect(df)
        
        # Sinais combinados
        df['Long_Signal'] = (
            (df['Stoch_Overbought']) &  # Estocástico sobrecomprado (sinal de compra)
            ((df['Respect_B1'] == 1) | (df['Respect_B2'] == 1) | (df['Respect_B3'] == 1)) &  # Respeitando barreira
            (df['RSI_Bull'])  # RSI > 50 (bull market)
        ).astype(int)
        
        df['Short_Signal'] = (
            (df['Stoch_Oversold']) &  # Estocástico sobrevendido (sinal de venda)
            ((df['Respect_B1'] == -1) | (df['Respect_B2'] == -1) | (df['Respect_B3'] == -1)) &  # Respeitando barreira
            (df['RSI_Bear'])  # RSI < 50 (bear market)
        ).astype(int)
        
        # Sinal final
        df['Final_Signal'] = np.where(df['Long_Signal'], 1, 
                                    np.where(df['Short_Signal'], -1, 0))
        
        return df
    
    def get_latest_analysis(self, data: pd.DataFrame) -> Dict:
        """Obtém a análise mais recente"""
        if data.empty:
            return {}
        
        latest = data.iloc[-1]
        return {
            'price': latest['Close'],
            'trend': 'Bull' if latest['Trend'] == 1 else 'Bear',
            'barreira_1': latest['Barreira_1'],
            'barreira_2': latest['Barreira_2'], 
            'barreira_3': latest['Barreira_3'],
            'rsi_75': latest['RSI_75'],
            'rsi_direction': 'Bull' if latest['RSI_Bull'] else 'Bear',
            'stoch_k': latest['Stoch_K'],
            'stoch_d': latest['Stoch_D'],
            'stoch_oversold': latest['Stoch_Oversold'],
            'stoch_overbought': latest['Stoch_Overbought'],
            'respect_b1': latest['Respect_B1'],
            'respect_b2': latest['Respect_B2'],
            'respect_b3': latest['Respect_B3'],
            'final_signal': latest['Final_Signal'],
            'long_signal': latest['Long_Signal'],
            'short_signal': latest['Short_Signal']
        }

def create_advanced_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Cria gráfico avançado com barreiras EMA"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{symbol} - Preço e Barreiras EMA',
            'RSI 75 Períodos',
            'Estocástico',
            'Sinais de Entrada'
        ),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Gráfico principal - Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Preço',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Barreiras EMA
    barrier_colors = ['orange', 'blue', 'red']
    barrier_names = ['Barreira 1 (Rápida)', 'Barreira 2 (Média)', 'Barreira 3 (Lenta)']
    
    for i, (barrier, color, name) in enumerate(zip(['Barreira_1', 'Barreira_2', 'Barreira_3'], 
                                                   barrier_colors, barrier_names)):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[barrier],
                name=name,
                line=dict(color=color, width=2)
            ),
            row=1, col=1
        )
    
    # Pontos de respeito às barreiras
    respect_data = data[(data['Respect_B1'] != 0) | (data['Respect_B2'] != 0) | (data['Respect_B3'] != 0)]
    if not respect_data.empty:
        fig.add_trace(
            go.Scatter(
                x=respect_data.index,
                y=respect_data['Close'],
                mode='markers',
                name='Respeito à Barreira',
                marker=dict(color='yellow', size=8, symbol='diamond')
            ),
            row=1, col=1
        )
    
    # RSI 75
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI_75'],
            name='RSI 75',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Linha de referência RSI 50
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_annotation(x=data.index[-1], y=55, text="Bull > 50", row=2, col=1, showarrow=False)
    fig.add_annotation(x=data.index[-1], y=45, text="Bear < 50", row=2, col=1, showarrow=False)
    
    # Estocástico
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Stoch_K'],
            name='Stoch %K',
            line=dict(color='blue')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Stoch_D'],
            name='Stoch %D', 
            line=dict(color='orange')
        ),
        row=3, col=1
    )
    
    # Linhas de referência Estocástico
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
    
    # Sinais de entrada
    long_signals = data[data['Long_Signal'] == 1]
    short_signals = data[data['Short_Signal'] == 1]
    
    if not long_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=long_signals.index,
                y=[1] * len(long_signals),
                mode='markers',
                name='Sinal LONG',
                marker=dict(color='green', size=12, symbol='triangle-up')
            ),
            row=4, col=1
        )
    
    if not short_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=short_signals.index,
                y=[-1] * len(short_signals),
                mode='markers',
                name='Sinal SHORT',
                marker=dict(color='red', size=12, symbol='triangle-down')
            ),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f'Análise de Barreiras EMA - {symbol}',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True
    )
    
    return fig

def main():
    st.title("₿ Analisador de Criptomoedas - Estratégia de Barreiras EMA")
    st.markdown("**Nova estratégia: Barreiras EMA + RSI 75 + Estocástico com Respeito às Barreiras**")
    
    # Inicializar analisador
    analyzer = KuCoinCryptoAnalyzer()
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção da criptomoeda
    selected_symbol = st.sidebar.selectbox(
        "Escolha a Criptomoeda:",
        options=analyzer.crypto_symbols,
        index=0
    )
    
    # Timeframe
    timeframe = st.sidebar.selectbox(
        "Timeframe:",
        options=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        index=6  # 1d como padrão
    )
    
    # Número de dias
    days = st.sidebar.number_input(
        "Dias de Histórico:",
        min_value=50,
        max_value=1000,
        value=200
    )
    
    # Mostrar configuração das barreiras
    st.sidebar.subheader("📊 Barreiras EMA Configuradas")
    if selected_symbol in EMA_BARRIERS:
        barriers = EMA_BARRIERS[selected_symbol]
        st.sidebar.write(f"**{selected_symbol}**")
        st.sidebar.write(f"• Barreira 1: {barriers['barreira_1']}")
        st.sidebar.write(f"• Barreira 2: {barriers['barreira_2']}")
        st.sidebar.write(f"• Barreira 3: {barriers['barreira_3']}")
    else:
        st.sidebar.write("Usando configurações padrão")
    
    # Botão de análise
    if st.sidebar.button("🚀 Analisar", type="primary"):
        with st.spinner(f"Buscando dados da KuCoin para {selected_symbol}..."):
            # Buscar dados
            data = analyzer.get_kucoin_data(selected_symbol, timeframe, days)
            
            if not data.empty:
                # Aplicar indicadores e estratégia
                data = analyzer.calculate_ema_barriers(data, selected_symbol)
                data = analyzer.calculate_rsi_75(data)
                data = analyzer.calculate_stochastic(data)
                data = analyzer.apply_strategy(data)
                
                # Análise atual
                analysis = analyzer.get_latest_analysis(data)
                
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Preço Atual",
                        f"${analysis['price']:.6f}",
                        delta=f"{((analysis['price'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%" if len(data) > 1 else "0%"
                    )
                
                with col2:
                    trend_color = "🟢" if analysis['trend'] == 'Bull' else "🔴"
                    st.metric("Tendência", f"{trend_color} {analysis['trend']}")
                
                with col3:
                    rsi_color = "🟢" if analysis['rsi_direction'] == 'Bull' else "🔴"
                    st.metric("RSI 75", f"{rsi_color} {analysis['rsi_75']:.1f}")
                
                with col4:
                    signal_text = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ NEUTRO"}
                    st.metric("Sinal", signal_text.get(analysis['final_signal'], "❓"))
                
                # Gráfico principal
                chart = create_advanced_chart(data, selected_symbol)
                st.plotly_chart(chart, use_container_width=True)
                
                # Análise detalhada
                st.subheader("📋 Análise Detalhada")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Barreiras EMA:**")
                    st.write(f"• Barreira 1: ${analysis['barreira_1']:.6f}")
                    st.write(f"• Barreira 2: ${analysis['barreira_2']:.6f}")
                    st.write(f"• Barreira 3: ${analysis['barreira_3']:.6f}")
                    
                    st.write("**Estocástico:**")
                    st.write(f"• %K: {analysis['stoch_k']:.1f}")
                    st.write(f"• %D: {analysis['stoch_d']:.1f}")
                    
                    oversold_text = "🔴 Sobrevendido" if analysis['stoch_oversold'] else ""
                    overbought_text = "🟢 Sobrecomprado" if analysis['stoch_overbought'] else ""
                    st.write(f"• Status: {oversold_text}{overbought_text}")
                
                with col2:
                    st.write("**Respeito às Barreiras:**")
                    respect_texts = []
                    if analysis['respect_b1'] != 0:
                        direction = "↑" if analysis['respect_b1'] > 0 else "↓"
                        respect_texts.append(f"• Barreira 1: {direction}")
                    if analysis['respect_b2'] != 0:
                        direction = "↑" if analysis['respect_b2'] > 0 else "↓"
                        respect_texts.append(f"• Barreira 2: {direction}")
                    if analysis['respect_b3'] != 0:
                        direction = "↑" if analysis['respect_b3'] > 0 else "↓"
                        respect_texts.append(f"• Barreira 3: {direction}")
                    
                    if respect_texts:
                        for text in respect_texts:
                            st.write(text)
                    else:
                        st.write("• Nenhuma barreira respeitada recentemente")
                    
                    st.write("**Condições de Entrada:**")
                    if analysis['long_signal']:
                        st.write("✅ **LONG**: Estoc sobrecomprado + Barreira respeitada + RSI Bull")
                    elif analysis['short_signal']:
                        st.write("✅ **SHORT**: Estoc sobrevendido + Barreira respeitada + RSI Bear")
                    else:
                        st.write("❌ Condições não atendidas")
                
                # Resumo da estratégia
                st.subheader("📖 Estratégia de Barreiras EMA")
                st.info("""
                **Como Funciona:**
                
                🎯 **Barreiras EMA**: Cada ativo tem 3 barreiras personalizadas (rápida, média, lenta)
                
                📈 **Tendência**: Definida pela Barreira 3 - mudança só quando preço ultrapassa completamente
                
                🔢 **RSI 75 Períodos**: > 50 = Bull Market | < 50 = Bear Market
                
                ⚡ **Entrada**: Estocástico + Respeito à Barreira + RSI na mesma direção
                
                🟢 **LONG**: Estocástico sobrecomprado (>80) + Toca e respeita barreira + RSI > 50
                
                🔴 **SHORT**: Estocástico sobrevendido (<20) + Toca e respeita barreira + RSI < 50
                """)
    
    # Informações adicionais
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Dados:** KuCoin API em tempo real")
    st.sidebar.markdown("**⚠️ Aviso:** Este não é um conselho financeiro.")

if __name__ == "__main__":
    main()
