import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from typing import Dict, Tuple, List

# Configuração da página
st.set_page_config(
    page_title="Analisador de Criptomoedas",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Classe principal do analisador
class CryptoAnalyzer:
    def __init__(self):
        self.crypto_symbols = {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD", 
            "Binance Coin": "BNB-USD",
            "Cardano": "ADA-USD",
            "Solana": "SOL-USD",
            "XRP": "XRP-USD",
            "Dogecoin": "DOGE-USD",
            "Polygon": "MATIC-USD",
            "Polkadot": "DOT-USD",
            "Avalanche": "AVAX-USD"
        }
        
    def get_crypto_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Busca dados da criptomoeda usando yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Erro ao buscar dados para {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_ema(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calcula múltiplas EMAs"""
        df = data.copy()
        for period in periods:
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(
                close=df['Close'], window=period
            ).ema_indicator()
        return df
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcula RSI"""
        df = data.copy()
        df['RSI'] = ta.momentum.RSIIndicator(
            close=df['Close'], window=period
        ).rsi()
        return df
    
    def calculate_stochastic_rsi(self, data: pd.DataFrame, 
                               k_period: int = 14, 
                               d_period: int = 3, 
                               rsi_period: int = 14) -> pd.DataFrame:
        """Calcula RSI Estocástico"""
        df = data.copy()
        
        # Primeiro calcula o RSI
        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_period).rsi()
        
        # Calcula o Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close=df['Close'], 
            window=rsi_period, 
            smooth1=k_period, 
            smooth2=d_period
        )
        
        df['StochRSI_K'] = stoch_rsi.stochrsi_k() * 100
        df['StochRSI_D'] = stoch_rsi.stochrsi_d() * 100
        
        return df
    
    def apply_strategy(self, data: pd.DataFrame, 
                      ema_fast: int, ema_slow: int,
                      rsi_oversold: int = 30, rsi_overbought: int = 70,
                      stoch_oversold: int = 20, stoch_overbought: int = 80) -> pd.DataFrame:
        """Aplica a estratégia de trading"""
        df = data.copy()
        
        # Sinais de EMA
        df['EMA_Signal'] = np.where(df[f'EMA_{ema_fast}'] > df[f'EMA_{ema_slow}'], 1, -1)
        
        # Sinais de RSI
        df['RSI_Signal'] = np.where(
            df['RSI'] < rsi_oversold, 1,  # Compra quando oversold
            np.where(df['RSI'] > rsi_overbought, -1, 0)  # Vende quando overbought
        )
        
        # Sinais de Stochastic RSI
        df['StochRSI_Signal'] = np.where(
            (df['StochRSI_K'] < stoch_oversold) & (df['StochRSI_D'] < stoch_oversold), 1,
            np.where(
                (df['StochRSI_K'] > stoch_overbought) & (df['StochRSI_D'] > stoch_overbought), -1, 0
            )
        )
        
        # Sinal combinado (estratégia)
        df['Combined_Signal'] = np.where(
            (df['EMA_Signal'] == 1) & 
            (df['RSI_Signal'] == 1) & 
            (df['StochRSI_Signal'] == 1), 1,  # COMPRA forte
            np.where(
                (df['EMA_Signal'] == -1) & 
                (df['RSI_Signal'] == -1) & 
                (df['StochRSI_Signal'] == -1), -1,  # VENDA forte
                np.where(
                    (df['EMA_Signal'] == 1) & 
                    ((df['RSI_Signal'] == 1) | (df['StochRSI_Signal'] == 1)), 0.5,  # COMPRA moderada
                    np.where(
                        (df['EMA_Signal'] == -1) & 
                        ((df['RSI_Signal'] == -1) | (df['StochRSI_Signal'] == -1)), -0.5, 0  # VENDA moderada
                    )
                )
            )
        )
        
        return df
    
    def get_latest_signals(self, data: pd.DataFrame) -> Dict:
        """Obtém os sinais mais recentes"""
        if data.empty:
            return {}
        
        latest = data.iloc[-1]
        return {
            'price': latest['Close'],
            'ema_signal': latest['EMA_Signal'],
            'rsi': latest['RSI'],
            'rsi_signal': latest['RSI_Signal'],
            'stoch_k': latest['StochRSI_K'],
            'stoch_d': latest['StochRSI_D'],
            'stoch_signal': latest['StochRSI_Signal'],
            'combined_signal': latest['Combined_Signal']
        }

def create_chart(data: pd.DataFrame, crypto_name: str, ema_periods: List[int]) -> go.Figure:
    """Cria gráfico interativo com Plotly"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{crypto_name} - Preço e EMAs',
            'RSI',
            'Stochastic RSI',
            'Sinais Combinados'
        ),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Gráfico de preço e EMAs
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Preço'
        ),
        row=1, col=1
    )
    
    # EMAs
    colors = ['orange', 'blue', 'green', 'red', 'purple']
    for i, period in enumerate(ema_periods):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f'EMA_{period}'],
                name=f'EMA {period}',
                line=dict(color=colors[i % len(colors)])
            ),
            row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Linhas de referência RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Stochastic RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['StochRSI_K'],
            name='Stoch RSI %K',
            line=dict(color='blue')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['StochRSI_D'],
            name='Stoch RSI %D',
            line=dict(color='orange')
        ),
        row=3, col=1
    )
    
    # Linhas de referência Stochastic RSI
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
    
    # Sinais combinados
    colors_signal = {1: 'green', 0.5: 'lightgreen', 0: 'gray', -0.5: 'orange', -1: 'red'}
    
    for signal_value, color in colors_signal.items():
        signal_data = data[data['Combined_Signal'] == signal_value]
        if not signal_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=signal_data.index,
                    y=[signal_value] * len(signal_data),
                    mode='markers',
                    name=f'Signal {signal_value}',
                    marker=dict(color=color, size=8),
                    showlegend=False
                ),
                row=4, col=1
            )
    
    fig.update_layout(
        title=f'Análise Técnica - {crypto_name}',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    st.title("₿ Analisador de Criptomoedas")
    st.markdown("**Análise técnica usando EMA, RSI e RSI Estocástico**")
    
    # Inicializar analisador
    analyzer = CryptoAnalyzer()
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção da criptomoeda
    crypto_name = st.sidebar.selectbox(
        "Escolha a Criptomoeda:",
        options=list(analyzer.crypto_symbols.keys()),
        index=0
    )
    
    # Período de dados
    period = st.sidebar.selectbox(
        "Período de Análise:",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    # Configurações de EMA
    st.sidebar.subheader("📈 Configurações EMA")
    ema_fast = st.sidebar.number_input("EMA Rápida", min_value=5, max_value=50, value=12)
    ema_slow = st.sidebar.number_input("EMA Lenta", min_value=20, max_value=200, value=26)
    ema_trend = st.sidebar.number_input("EMA Tendência", min_value=50, max_value=300, value=200)
    
    # Configurações RSI
    st.sidebar.subheader("📊 Configurações RSI")
    rsi_period = st.sidebar.number_input("Período RSI", min_value=10, max_value=30, value=14)
    rsi_oversold = st.sidebar.number_input("RSI Sobrevenda", min_value=20, max_value=40, value=30)
    rsi_overbought = st.sidebar.number_input("RSI Sobrecompra", min_value=60, max_value=80, value=70)
    
    # Configurações Stochastic RSI
    st.sidebar.subheader("📉 Configurações Stoch RSI")
    stoch_k = st.sidebar.number_input("Período %K", min_value=10, max_value=20, value=14)
    stoch_d = st.sidebar.number_input("Período %D", min_value=3, max_value=10, value=3)
    stoch_oversold = st.sidebar.number_input("Stoch Sobrevenda", min_value=10, max_value=30, value=20)
    stoch_overbought = st.sidebar.number_input("Stoch Sobrecompra", min_value=70, max_value=90, value=80)
    
    # Botão para análise
    if st.sidebar.button("🚀 Analisar", type="primary"):
        with st.spinner(f"Analisando {crypto_name}..."):
            # Buscar dados
            symbol = analyzer.crypto_symbols[crypto_name]
            data = analyzer.get_crypto_data(symbol, period)
            
            if not data.empty:
                # Calcular indicadores
                ema_periods = [ema_fast, ema_slow, ema_trend]
                data = analyzer.calculate_ema(data, ema_periods)
                data = analyzer.calculate_rsi(data, rsi_period)
                data = analyzer.calculate_stochastic_rsi(data, stoch_k, stoch_d, rsi_period)
                data = analyzer.apply_strategy(
                    data, ema_fast, ema_slow, 
                    rsi_oversold, rsi_overbought,
                    stoch_oversold, stoch_overbought
                )
                
                # Obter sinais atuais
                signals = analyzer.get_latest_signals(data)
                
                # Layout principal
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Preço Atual",
                        f"${signals['price']:.2f}",
                        delta=f"{((signals['price'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%"
                    )
                
                with col2:
                    st.metric("RSI", f"{signals['rsi']:.1f}")
                
                with col3:
                    signal_text = {
                        1: "🟢 COMPRA FORTE",
                        0.5: "🟡 COMPRA MODERADA", 
                        0: "⚪ NEUTRO",
                        -0.5: "🟠 VENDA MODERADA",
                        -1: "🔴 VENDA FORTE"
                    }
                    st.metric("Sinal", signal_text.get(signals['combined_signal'], "❓"))
                
                # Gráfico principal
                chart = create_chart(data, crypto_name, ema_periods)
                st.plotly_chart(chart, use_container_width=True)
                
                # Análise detalhada
                st.subheader("📋 Análise Detalhada")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Indicadores Atuais:**")
                    st.write(f"• EMA {ema_fast}: ${data[f'EMA_{ema_fast}'].iloc[-1]:.2f}")
                    st.write(f"• EMA {ema_slow}: ${data[f'EMA_{ema_slow}'].iloc[-1]:.2f}")
                    st.write(f"• EMA {ema_trend}: ${data[f'EMA_{ema_trend}'].iloc[-1]:.2f}")
                    st.write(f"• RSI: {signals['rsi']:.1f}")
                    st.write(f"• Stoch RSI %K: {signals['stoch_k']:.1f}")
                    st.write(f"• Stoch RSI %D: {signals['stoch_d']:.1f}")
                
                with col2:
                    st.write("**Interpretação:**")
                    
                    # EMA
                    if signals['ema_signal'] == 1:
                        st.write("🟢 **Tendência:** Alta (EMA rápida > EMA lenta)")
                    else:
                        st.write("🔴 **Tendência:** Baixa (EMA rápida < EMA lenta)")
                    
                    # RSI
                    if signals['rsi'] < rsi_oversold:
                        st.write("🟢 **RSI:** Sobrevenda - Possível compra")
                    elif signals['rsi'] > rsi_overbought:
                        st.write("🔴 **RSI:** Sobrecompra - Possível venda")
                    else:
                        st.write("🟡 **RSI:** Zona neutra")
                    
                    # Stoch RSI
                    if signals['stoch_k'] < stoch_oversold:
                        st.write("🟢 **Stoch RSI:** Sobrevenda")
                    elif signals['stoch_k'] > stoch_overbought:
                        st.write("🔴 **Stoch RSI:** Sobrecompra")
                    else:
                        st.write("🟡 **Stoch RSI:** Zona neutra")
                
                # Resumo da estratégia
                st.subheader("📖 Resumo da Estratégia")
                st.info("""
                **Estratégia Utilizada:**
                
                🟢 **COMPRA FORTE**: EMA rápida > EMA lenta + RSI em sobrevenda + Stoch RSI em sobrevenda
                
                🟡 **COMPRA MODERADA**: EMA rápida > EMA lenta + (RSI em sobrevenda OU Stoch RSI em sobrevenda)
                
                🔴 **VENDA FORTE**: EMA rápida < EMA lenta + RSI em sobrecompra + Stoch RSI em sobrecompra
                
                🟠 **VENDA MODERADA**: EMA rápida < EMA lenta + (RSI em sobrecompra OU Stoch RSI em sobrecompra)
                
                ⚪ **NEUTRO**: Demais condições
                """)
    
    # Informações adicionais
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Dica:** Ajuste os parâmetros conforme sua estratégia de trading.")
    st.sidebar.markdown("**⚠️ Aviso:** Este não é um conselho financeiro. Sempre faça sua própria pesquisa.")

if __name__ == "__main__":
    main()