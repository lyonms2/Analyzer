def create_advanced_chart(data: pd.DataFrame, symbol: str) -> go.Figure:import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from typing import Dict, Tuple, List
import time
import pytz
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
        self.brazil_tz = pytz.timezone('America/Sao_Paulo')
        
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
            # Converter para fuso horário do Brasil
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(self.brazil_tz)
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
    
    def calculate_stochastic_rsi(self, data: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
        """Calcula Stochastic RSI igual ao TradingView"""
        df = data.copy()
        
        # Primeiro calcular o RSI
        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_period).rsi()
        
        # Calcular o Stochastic do RSI manualmente para ficar igual ao TradingView
        stoch_rsi_values = []
        
        for i in range(len(rsi)):
            if i < stoch_period - 1:
                stoch_rsi_values.append(np.nan)
            else:
                # Pegar os últimos stoch_period valores do RSI
                rsi_period_values = rsi.iloc[i-stoch_period+1:i+1]
                
                if rsi_period_values.isna().any():
                    stoch_rsi_values.append(np.nan)
                else:
                    # Calcular Stochastic RSI
                    rsi_min = rsi_period_values.min()
                    rsi_max = rsi_period_values.max()
                    
                    if rsi_max - rsi_min == 0:
                        stoch_rsi_values.append(50)  # Valor neutro quando não há variação
                    else:
                        stoch_rsi = ((rsi.iloc[i] - rsi_min) / (rsi_max - rsi_min)) * 100
                        stoch_rsi_values.append(stoch_rsi)
        
        # Aplicar suavização K
        stoch_rsi_series = pd.Series(stoch_rsi_values, index=df.index)
        df['StochRSI_Raw'] = stoch_rsi_series
        
        # %K = SMA do StochRSI Raw
        df['Stoch_K'] = df['StochRSI_Raw'].rolling(window=k_smooth, min_periods=1).mean()
        
        # %D = SMA do %K
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_smooth, min_periods=1).mean()
        
        # Remover coluna temporária
        df.drop('StochRSI_Raw', axis=1, inplace=True)
        
        return df
    
    def detect_barrier_breakthrough(self, data: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
        """Detecta quando o preço vem de baixo e fecha acima da barreira (compra) 
        ou vem de cima e fecha abaixo da barreira (venda)"""
        df = data.copy()
        
        # Inicializar colunas de breakthrough das barreiras
        df['Breakthrough_B1'] = 0
        df['Breakthrough_B2'] = 0  
        df['Breakthrough_B3'] = 0
        
        for i in range(lookback, len(df)):
            current_close = df['Close'].iloc[i]
            previous_closes = df['Close'].iloc[i-lookback:i]
            
            # Verificar cada barreira
            barriers = ['Barreira_1', 'Barreira_2', 'Barreira_3']
            breakthrough_cols = ['Breakthrough_B1', 'Breakthrough_B2', 'Breakthrough_B3']
            
            for barrier, breakthrough_col in zip(barriers, breakthrough_cols):
                barrier_value = df[barrier].iloc[i]
                
                if pd.isna(barrier_value):
                    continue
                
                # Verificar breakthrough de baixo para cima (sinal de compra)
                # Condições: preços anteriores estavam abaixo da barreira E preço atual fechou acima
                was_below = any(prev_close < barrier_value * 0.999 for prev_close in previous_closes)  # 0.1% tolerance
                closed_above = current_close > barrier_value * 1.001  # 0.1% tolerance
                
                if was_below and closed_above:
                    df.loc[df.index[i], breakthrough_col] = 1  # Sinal de compra
                    continue
                
                # Verificar breakthrough de cima para baixo (sinal de venda)
                # Condições: preços anteriores estavam acima da barreira E preço atual fechou abaixo
                was_above = any(prev_close > barrier_value * 1.001 for prev_close in previous_closes)
                closed_below = current_close < barrier_value * 0.999
                
                if was_above and closed_below:
                    df.loc[df.index[i], breakthrough_col] = -1  # Sinal de venda
        
        return df
    
    def apply_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica a estratégia corrigida com breakthrough das barreiras"""
        df = data.copy()
        
        # Determinar tendência geral (Barreira 3)
        df['Trend'] = np.where(df['Close'] > df['Barreira_3'], 1, -1)  # 1=Bull, -1=Bear
        
        # Sinais RSI (75 períodos)
        df['RSI_Bull'] = df['RSI_75'] > 50
        df['RSI_Bear'] = df['RSI_75'] < 50
        
        # Sinais Estocástico RSI (CORRIGIDOS)
        df['Stoch_Oversold'] = df['Stoch_K'] < 20   # Para COMPRA
        df['Stoch_Overbought'] = df['Stoch_K'] > 80  # Para VENDA
        
        # Detectar breakthrough das barreiras
        df = self.detect_barrier_breakthrough(df)
        
        # Sinais combinados CORRIGIDOS
        df['Long_Signal'] = (
            (df['RSI_Bull']) &  # RSI > 50 (bull market)
            ((df['Breakthrough_B1'] == 1) | (df['Breakthrough_B2'] == 1) | (df['Breakthrough_B3'] == 1)) &  # Preço veio de baixo e fechou acima da barreira
            (df['Stoch_Oversold'])  # StochRSI < 20 (sobrevendido)
        ).astype(int)
        
        df['Short_Signal'] = (
            (df['RSI_Bear']) &  # RSI < 50 (bear market)
            ((df['Breakthrough_B1'] == -1) | (df['Breakthrough_B2'] == -1) | (df['Breakthrough_B3'] == -1)) &  # Preço veio de cima e fechou abaixo da barreira
            (df['Stoch_Overbought'])  # StochRSI > 80 (sobrecomprado)
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
            'respect_b1': latest['Breakthrough_B1'],
            'respect_b2': latest['Breakthrough_B2'],
            'respect_b3': latest['Breakthrough_B3'],
            'final_signal': latest['Final_Signal'],
            'long_signal': latest['Long_Signal'],
            'short_signal': latest['Short_Signal']
        }
    
    def scan_all_cryptos(self, timeframe: str = "1d", days: int = 100) -> Dict[str, List]:
        """Escaneia todas as criptomoedas em busca de sinais de compra e venda"""
        buy_signals = []
        sell_signals = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_cryptos = len(self.crypto_symbols)
        
        for i, symbol in enumerate(self.crypto_symbols):
            try:
                # Atualizar progresso
                progress = (i + 1) / total_cryptos
                progress_bar.progress(progress)
                status_text.text(f"Analisando {symbol}... ({i+1}/{total_cryptos})")
                
                # Buscar e analisar dados
                data = self.get_kucoin_data(symbol, timeframe, days)
                
                if not data.empty:
                    # Aplicar indicadores
                    data = self.calculate_ema_barriers(data, symbol)
                    data = self.calculate_rsi_75(data)
                    data = self.calculate_stochastic_rsi(data)
                    data = self.apply_strategy(data)
                    
                    # Verificar sinais mais recentes
                    analysis = self.get_latest_analysis(data)
                    
                    if analysis.get('long_signal', 0) == 1:
                        buy_signals.append({
                            'symbol': symbol,
                            'price': analysis['price'],
                            'rsi_75': analysis['rsi_75'],
                            'stoch_k': analysis['stoch_k'],
                            'breakthrough_b1': analysis['respect_b1'],
                            'breakthrough_b2': analysis['respect_b2'],
                            'breakthrough_b3': analysis['respect_b3'],
                            'timestamp': data.index[-1].strftime('%d/%m/%Y %H:%M')
                        })
                    
                    if analysis.get('short_signal', 0) == 1:
                        sell_signals.append({
                            'symbol': symbol,
                            'price': analysis['price'],
                            'rsi_75': analysis['rsi_75'],
                            'stoch_k': analysis['stoch_k'],
                            'breakthrough_b1': analysis['respect_b1'],
                            'breakthrough_b2': analysis['respect_b2'],
                            'breakthrough_b3': analysis['respect_b3'],
                            'timestamp': data.index[-1].strftime('%d/%m/%Y %H:%M')
                        })
                
            except Exception as e:
                errors.append(f"{symbol}: {str(e)[:50]}")
                continue
        
        # Limpar barra de progresso
        progress_bar.empty()
        status_text.empty()
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'errors': errors
        }

def display_signals_table(signals_data: Dict, signal_type: str):
    """Exibe tabela com sinais encontrados"""
    if signal_type == "buy":
        signals = signals_data['buy_signals']
        title = "🟢 OPORTUNIDADES DE COMPRA"
        color = "green"
    else:
        signals = signals_data['sell_signals']
        title = "🔴 OPORTUNIDADES DE VENDA"
        color = "red"
    
    if not signals:
        st.info(f"Nenhum sinal de {signal_type.upper()} encontrado no momento.")
        return
    
    st.markdown(f"### {title} ({len(signals)} encontradas)")
    
    # Converter para DataFrame
    df = pd.DataFrame(signals)
    
    # Formatação das colunas
    df['price'] = df['price'].apply(lambda x: f"${x:.6f}")
    df['rsi_75'] = df['rsi_75'].apply(lambda x: f"{x:.1f}")
    df['stoch_k'] = df['stoch_k'].apply(lambda x: f"{x:.1f}")
    
    # Identificar qual barreira foi rompida
    def get_breakthrough_info(row):
        breakthroughs = []
        if row['breakthrough_b1'] != 0:
            direction = "↗️" if row['breakthrough_b1'] > 0 else "↘️"
            breakthroughs.append(f"B1{direction}")
        if row['breakthrough_b2'] != 0:
            direction = "↗️" if row['breakthrough_b2'] > 0 else "↘️"
            breakthroughs.append(f"B2{direction}")
        if row['breakthrough_b3'] != 0:
            direction = "↗️" if row['breakthrough_b3'] > 0 else "↘️"
            breakthroughs.append(f"B3{direction}")
        return " | ".join(breakthroughs) if breakthroughs else "N/A"
    
    df['barreiras'] = df.apply(get_breakthrough_info, axis=1)
    
    # Reorganizar colunas para exibição
    display_df = df[['symbol', 'price', 'rsi_75', 'stoch_k', 'barreiras', 'timestamp']].copy()
    display_df.columns = ['Moeda', 'Preço', 'RSI 75', 'Stoch K', 'Barreiras', 'Horário (BR)']
    
    # Exibir tabela
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Estatísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Sinais", len(signals))
    with col2:
        avg_rsi = np.mean([s['rsi_75'] for s in signals])
        st.metric("RSI Médio", f"{avg_rsi:.1f}")
    with col3:
        avg_stoch = np.mean([s['stoch_k'] for s in signals])
        st.metric("Stoch K Médio", f"{avg_stoch:.1f}")
    """Cria gráfico avançado com barreiras EMA"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{symbol} - Preço e Barreiras EMA',
            'RSI 75 Períodos',
            'Estocástico RSI',
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
    
    # Pontos de breakthrough das barreiras
    breakthrough_data = data[(data['Breakthrough_B1'] != 0) | (data['Breakthrough_B2'] != 0) | (data['Breakthrough_B3'] != 0)]
    if not breakthrough_data.empty:
        # Separar compras (breakthrough positivo) e vendas (breakthrough negativo)
        buy_breakthroughs = breakthrough_data[
            (breakthrough_data['Breakthrough_B1'] == 1) | 
            (breakthrough_data['Breakthrough_B2'] == 1) | 
            (breakthrough_data['Breakthrough_B3'] == 1)
        ]
        sell_breakthroughs = breakthrough_data[
            (breakthrough_data['Breakthrough_B1'] == -1) | 
            (breakthrough_data['Breakthrough_B2'] == -1) | 
            (breakthrough_data['Breakthrough_B3'] == -1)
        ]
        
        if not buy_breakthroughs.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_breakthroughs.index,
                    y=buy_breakthroughs['Close'],
                    mode='markers',
                    name='Breakthrough Compra',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        if not sell_breakthroughs.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_breakthroughs.index,
                    y=sell_breakthroughs['Close'],
                    mode='markers',
                    name='Breakthrough Venda',
                    marker=dict(color='red', size=8, symbol='triangle-down')
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
    
    # Estocástico RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Stoch_K'],
            name='StochRSI %K',
            line=dict(color='blue')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Stoch_D'],
            name='StochRSI %D', 
            line=dict(color='orange')
        ),
        row=3, col=1
    )
    
    # Linhas de referência Stochastic RSI
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
    st.markdown("**Nova estratégia: Barreiras EMA + RSI 75 + Estocástico RSI com Breakthrough das Barreiras**")
    
    # Mostrar horário atual do Brasil
    brazil_tz = pytz.timezone('America/Sao_Paulo')
    current_time = datetime.now(brazil_tz)
    st.sidebar.markdown(f"🕐 **Horário BR:** {current_time.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Inicializar analisador
    analyzer = KuCoinCryptoAnalyzer()
    
    # Criar abas
    tab1, tab2 = st.tabs(["📊 Análise Individual", "🔍 Scanner Geral"])
    
    with tab1:
        # Sidebar para análise individual
        st.sidebar.header("⚙️ Análise Individual")
        
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
        
        # Botão de análise individual
        if st.sidebar.button("🚀 Analisar", type="primary"):
            with st.spinner(f"Buscando dados da KuCoin para {selected_symbol}..."):
                # Buscar dados
                data = analyzer.get_kucoin_data(selected_symbol, timeframe, days)
                
                if not data.empty:
                    # Aplicar indicadores e estratégia
                    data = analyzer.calculate_ema_barriers(data, selected_symbol)
                    data = analyzer.calculate_rsi_75(data)
                    data = analyzer.calculate_stochastic_rsi(data)  # Usando Stochastic RSI agora
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
                        
                        st.write("**Estocástico RSI:**")
                        st.write(f"• %K: {analysis['stoch_k']:.1f}")
                        st.write(f"• %D: {analysis['stoch_d']:.1f}")
                        
                        oversold_text = "🟢 Sobrevendido" if analysis['stoch_oversold'] else ""
                        overbought_text = "🔴 Sobrecomprado" if analysis['stoch_overbought'] else ""
                        st.write(f"• Status: {oversold_text}{overbought_text}")
                    
                    with col2:
                        st.write("**Breakthrough das Barreiras:**")
                        breakthrough_texts = []
                        if analysis['respect_b1'] == 1:
                            breakthrough_texts.append("• Barreira 1: ↗️ Rompeu de baixo")
                        elif analysis['respect_b1'] == -1:
                            breakthrough_texts.append("• Barreira 1: ↘️ Rompeu de cima")
                        
                        if analysis['respect_b2'] == 1:
                            breakthrough_texts.append("• Barreira 2: ↗️ Rompeu de baixo")
                        elif analysis['respect_b2'] == -1:
                            breakthrough_texts.append("• Barreira 2: ↘️ Rompeu de cima")
                            
                        if analysis['respect_b3'] == 1:
                            breakthrough_texts.append("• Barreira 3: ↗️ Rompeu de baixo")
                        elif analysis['respect_b3'] == -1:
                            breakthrough_texts.append("• Barreira 3: ↘️ Rompeu de cima")
                        
                        if breakthrough_texts:
                            for text in breakthrough_texts:
                                st.write(text)
                        else:
                            st.write("• Nenhuma barreira rompida recentemente")
                        
                        st.write("**Condições de Entrada:**")
                        if analysis['long_signal']:
                            st.write("✅ **LONG**: RSI > 50 + Breakthrough de baixo + StochRSI < 20")
                        elif analysis['short_signal']:
                            st.write("✅ **SHORT**: RSI < 50 + Breakthrough de cima + StochRSI > 80")
                        else:
                            st.write("❌ Condições não atendidas")
                    
                    # Resumo da estratégia
                    st.subheader("📖 Estratégia de Barreiras EMA")
                    st.info("""
                    **Como Funciona:**
                    
                    🎯 **Barreiras EMA**: Cada ativo tem 3 barreiras personalizadas (rápida, média, lenta)
                    
                    📈 **Tendência**: Definida pela Barreira 3 - mudança só quando preço ultrapassa completamente
                    
                    🔢 **RSI 75 Períodos**: > 50 = Bull Market | < 50 = Bear Market
                    
                    ⚡ **Entrada**: Breakthrough da Barreira + StochRSI + RSI na mesma direção
                    
                    🟢 **LONG**: RSI > 50 + Preço rompeu barreira de baixo para cima + StochRSI < 20
                    
                    🔴 **SHORT**: RSI < 50 + Preço rompeu barreira de cima para baixo + StochRSI > 80
                    """)
    
    with tab2:
        st.header("🔍 Scanner de Todas as Criptomoedas")
        st.markdown("Escaneia todas as 65 criptomoedas em busca de sinais de compra e venda.")
        
        # Configurações do scanner
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scanner_timeframe = st.selectbox(
                "Timeframe para Scanner:",
                options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=6,
                key="scanner_tf"
            )
        
        with col2:
            scanner_days = st.number_input(
                "Dias de Histórico:",
                min_value=50,
                max_value=500,
                value=100,
                key="scanner_days"
            )
        
        with col3:
            st.write("") # Espaço
            scan_button = st.button("🚀 Escanear Todas as Moedas", type="primary", key="scan_all")
        
        if scan_button:
            st.markdown("---")
            
            with st.spinner("🔍 Escaneando todas as criptomoedas..."):
                # Executar scanner
                signals_data = analyzer.scan_all_cryptos(scanner_timeframe, scanner_days)
                
                # Exibir resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    display_signals_table(signals_data, "buy")
                
                with col2:
                    display_signals_table(signals_data, "sell")
                
                # Resumo geral
                st.markdown("---")
                st.subheader("📊 Resumo do Scanner")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Analisadas", len(analyzer.crypto_symbols))
                
                with col2:
                    st.metric("Sinais de Compra", len(signals_data['buy_signals']))
                
                with col3:
                    st.metric("Sinais de Venda", len(signals_data['sell_signals']))
                
                with col4:
                    total_signals = len(signals_data['buy_signals']) + len(signals_data['sell_signals'])
                    success_rate = (total_signals / len(analyzer.crypto_symbols)) * 100
                    st.metric("Taxa de Sinais", f"{success_rate:.1f}%")
                
                # Mostrar erros se houver
                if signals_data['errors']:
                    st.warning(f"⚠️ Erros encontrados em {len(signals_data['errors'])} moedas:")
                    with st.expander("Ver erros"):
                        for error in signals_data['errors']:
                            st.write(f"• {error}")
        
        # Informações sobre o scanner
        st.markdown("---")
        st.info("""
        **ℹ️ Como usar o Scanner:**
        
        • **Timeframes menores** (1m, 5m): Sinais mais frequentes, mas menos confiáveis
        • **Timeframes maiores** (1d): Sinais mais confiáveis, mas menos frequentes
        • **Dias de Histórico**: Mais dias = indicadores mais estáveis
        • **Atualização**: Execute o scanner regularmente para capturar novos sinais
        """)
    
    # Informações adicionais
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Dados:** KuCoin API em tempo real")
    st.sidebar.markdown(f"**🌍 Fuso:** Brasil (UTC-3)")
    st.sidebar.markdown("**⚠️ Aviso:** Este não é um conselho financeiro.")
                **Como Funciona:**
                
                🎯 **Barreiras EMA**: Cada ativo tem 3 barreiras personalizadas (rápida, média, lenta)
                
                📈 **Tendência**: Definida pela Barreira 3 - mudança só quando preço ultrapassa completamente
                
                🔢 **RSI 75 Períodos**: > 50 = Bull Market | < 50 = Bear Market
                
                ⚡ **Entrada**: Breakthrough da Barreira + StochRSI + RSI na mesma direção
                
                🟢 **LONG**: RSI > 50 + Preço rompeu barreira de baixo para cima + StochRSI < 20
                
                🔴 **SHORT**: RSI < 50 + Preço rompeu barreira de cima para baixo + StochRSI > 80
                """)
    
    # Informações adicionais
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Dados:** KuCoin API em tempo real")
    st.sidebar.markdown("**⚠️ Aviso:** Este não é um conselho financeiro.")

if __name__ == "__main__":
    main()
