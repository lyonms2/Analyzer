import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
from typing import Dict, List
import time
import pytz
from datetime import datetime
from ema_barriers_config import EMA_BARRIERS, CRYPTO_LIST

# Configuração da página
st.set_page_config(
    page_title="Analisador Simplificado - Barreiras EMA",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class SimplifiedCryptoAnalyzer:
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.crypto_symbols = CRYPTO_LIST
        self.brazil_tz = pytz.timezone('America/Sao_Paulo')
        
    def get_kucoin_data(self, symbol: str, timeframe: str = "1day", days: int = 100) -> pd.DataFrame:
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
                return pd.DataFrame()
            
            # Converter para DataFrame
            df = pd.DataFrame(data['data'], columns=[
                'timestamp', 'Open', 'Close', 'High', 'Low', 'Volume', 'Turnover'
            ])
            
            # Processar dados
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(self.brazil_tz)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Converter para float
            numeric_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            return pd.DataFrame()
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calcula todos os indicadores necessários"""
        if data.empty or len(data) < 100:
            return {}
        
        try:
            # RSI 75 períodos
            rsi_75 = ta.momentum.RSIIndicator(close=data['Close'], window=75).rsi().iloc[-1]
            rsi_status = "Bull" if rsi_75 > 50 else "Bear"
            
            # Configurações EMA para o símbolo
            if symbol in EMA_BARRIERS:
                b1_period = EMA_BARRIERS[symbol]['barreira_1']
                b2_period = EMA_BARRIERS[symbol]['barreira_2']
                b3_period = EMA_BARRIERS[symbol]['barreira_3']
            else:
                b1_period, b2_period, b3_period = 21, 50, 200
            
            # Calcular EMAs
            ema_b1 = ta.trend.EMAIndicator(close=data['Close'], window=b1_period).ema_indicator()
            ema_b2 = ta.trend.EMAIndicator(close=data['Close'], window=b2_period).ema_indicator()
            ema_b3 = ta.trend.EMAIndicator(close=data['Close'], window=b3_period).ema_indicator()
            
            # Preços atual e anterior
            current_close = data['Close'].iloc[-1]
            previous_close = data['Close'].iloc[-2]
            
            # Valores das barreiras atual e anterior
            b1_current = ema_b1.iloc[-1]
            b1_previous = ema_b1.iloc[-2]
            b2_current = ema_b2.iloc[-1]
            b2_previous = ema_b2.iloc[-2]
            b3_current = ema_b3.iloc[-1]
            b3_previous = ema_b3.iloc[-2]
            
            # Analisar cada barreira
            def analyze_barrier(current_close, previous_close, barrier_current, barrier_previous):
                # Verificar cruzamento
                if previous_close < barrier_previous and current_close > barrier_current:
                    return "🟢"  # Cruzou de baixo para cima (compra)
                elif previous_close > barrier_previous and current_close < barrier_current:
                    return "🔴"  # Cruzou de cima para baixo (venda)
                else:
                    # Calcular distância percentual
                    distance = ((current_close - barrier_current) / barrier_current) * 100
                    return f"{distance:+.2f}%"
            
            b1_status = analyze_barrier(current_close, previous_close, b1_current, b1_previous)
            b2_status = analyze_barrier(current_close, previous_close, b2_current, b2_previous)
            b3_status = analyze_barrier(current_close, previous_close, b3_current, b3_previous)
            
            # Estocástico RSI com direção e cruzamentos
            rsi_14 = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
            
            # Calcular Stochastic RSI manualmente
            stoch_period = 14
            stoch_rsi_values = []
            
            for i in range(len(rsi_14)):
                if i < stoch_period - 1:
                    stoch_rsi_values.append(np.nan)
                else:
                    rsi_period_values = rsi_14.iloc[i-stoch_period+1:i+1]
                    if rsi_period_values.isna().any():
                        stoch_rsi_values.append(np.nan)
                    else:
                        rsi_min = rsi_period_values.min()
                        rsi_max = rsi_period_values.max()
                        if rsi_max - rsi_min == 0:
                            stoch_rsi_values.append(50)
                        else:
                            stoch_rsi = ((rsi_14.iloc[i] - rsi_min) / (rsi_max - rsi_min)) * 100
                            stoch_rsi_values.append(stoch_rsi)
            
            # Suavização K e D
            stoch_rsi_series = pd.Series(stoch_rsi_values, index=data.index)
            stoch_k = stoch_rsi_series.rolling(window=3, min_periods=1).mean()
            stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()
            
            # Valores atuais e anteriores
            k_current = stoch_k.iloc[-1]
            k_previous = stoch_k.iloc[-2]
            d_current = stoch_d.iloc[-1]
            d_previous = stoch_d.iloc[-2]
            
            # Detectar direção e cruzamentos
            def analyze_stochastic(k_curr, k_prev, d_curr, d_prev):
                # Detectar cruzamentos
                k_cross_d_up = k_prev <= d_prev and k_curr > d_curr  # K cruzou D de baixo para cima
                k_cross_d_down = k_prev >= d_prev and k_curr < d_curr  # K cruzou D de cima para baixo
                
                # Detectar direção
                k_diff = k_curr - k_prev
                is_rising = k_diff > 0.5
                is_falling = k_diff < -0.5
                is_sideways = abs(k_diff) <= 0.5
                
                # Determinar zona
                in_oversold = k_curr < 20
                in_overbought = k_curr > 80
                in_neutral = 20 <= k_curr <= 80
                
                # Criar símbolo visual
                if in_oversold:
                    if k_cross_d_up:
                        return f"⚡🚀 {k_curr:.1f}"  # SETUP PERFEITO compra
                    elif is_rising:
                        return f"🟢📈 {k_curr:.1f}"  # Sobrevenda subindo
                    elif is_falling:
                        return f"🟢📉 {k_curr:.1f}"  # Sobrevenda descendo
                    else:
                        return f"🟢➡️ {k_curr:.1f}"   # Sobrevenda lateral
                        
                elif in_overbought:
                    if k_cross_d_down:
                        return f"💀💥 {k_curr:.1f}"  # SETUP PERFEITO venda
                    elif is_falling:
                        return f"🔴📉 {k_curr:.1f}"  # Sobrecompra descendo
                    elif is_rising:
                        return f"🔴📈 {k_curr:.1f}"  # Sobrecompra subindo
                    else:
                        return f"🔴➡️ {k_curr:.1f}"   # Sobrecompra lateral
                        
                else:  # Zona neutra
                    if k_cross_d_up:
                        return f"🚀 {k_curr:.1f}"     # Cruzamento alta
                    elif k_cross_d_down:
                        return f"💥 {k_curr:.1f}"     # Cruzamento baixa
                    elif is_rising:
                        return f"📈 {k_curr:.1f}"     # Subindo
                    elif is_falling:
                        return f"📉 {k_curr:.1f}"     # Descendo
                    else:
                        return f"➡️ {k_curr:.1f}"      # Lateral
            
            stoch_visual = analyze_stochastic(k_current, k_previous, d_current, d_previous)
            
            # Mean Reversion Analysis - Distância da Média
            # Calcular média móvel de 20 períodos e desvio padrão
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            std_20 = data['Close'].rolling(window=20).std().iloc[-1]
            
            # Z-Score (distância em desvios padrão)
            z_score = (current_close - sma_20) / std_20
            
            # Distância percentual da média
            distance_pct = ((current_close - sma_20) / sma_20) * 100
            
            # RSI para confluência de extremos
            rsi_14_current = rsi_14.iloc[-1]
            
            # Bollinger Bands
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            
            def analyze_mean_reversion(price, sma, z_score, distance_pct, rsi_14, bb_upper, bb_lower):
                """Analisa oportunidades de mean reversion"""
                
                # Extremos matematicamente definidos
                extreme_oversold = z_score < -2.0 and rsi_14 < 25  # Fundo extremo
                extreme_overbought = z_score > 2.0 and rsi_14 > 75  # Topo extremo
                
                # Oportunidades de mean reversion
                oversold_opportunity = z_score < -1.5 and distance_pct < -10  # Oportunidade de compra
                overbought_opportunity = z_score > 1.5 and distance_pct > 10   # Oportunidade de venda
                
                # Zona de breakout (próximo da média)
                breakout_zone = abs(z_score) < 0.5 and abs(distance_pct) < 5
                
                # Estados visuais
                if extreme_oversold and price < bb_lower:
                    return f"🔥💚 {distance_pct:+.1f}%"  # EXTREME BUY - fundo perfeito
                elif extreme_overbought and price > bb_upper:
                    return f"🔥❤️ {distance_pct:+.1f}%"  # EXTREME SELL - topo perfeito
                elif oversold_opportunity:
                    return f"🟢📈 {distance_pct:+.1f}%"  # OVERSOLD - oportunidade compra
                elif overbought_opportunity:
                    return f"🔴📉 {distance_pct:+.1f}%"  # OVERBOUGHT - oportunidade venda
                elif breakout_zone:
                    return f"🔵⚡ {distance_pct:+.1f}%"   # BREAKOUT ZONE - usar EMAs
                else:
                    # Zona neutra com direção
                    if distance_pct > 0:
                        return f"⚪↗️ {distance_pct:+.1f}%"  # Acima da média
                    else:
                        return f"⚪↘️ {distance_pct:+.1f}%"  # Abaixo da média
            
            mean_reversion_signal = analyze_mean_reversion(
                current_close, sma_20, z_score, distance_pct, rsi_14_current, bb_upper, bb_lower
            )
            
            return {
                'symbol': symbol,
                'price': current_close,
                'rsi': f"{rsi_75:.1f} ({rsi_status})",
                'b1': b1_status,
                'b2': b2_status,
                'b3': b3_status,
                'stochastic': stoch_visual,
                'mean_reversion': mean_reversion_signal,
                'timestamp': data.index[-1].strftime('%H:%M')
            }
            
        except Exception as e:
            return {}
    
    def scan_all_cryptos(self, timeframe: str = "1d", days: int = 100) -> List[Dict]:
        """Escaneia todas as criptomoedas"""
        results = []
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
                    analysis = self.calculate_indicators(data, symbol)
                    if analysis:
                        results.append(analysis)
                
            except Exception as e:
                continue
        
        # Limpar barra de progresso
        progress_bar.empty()
        status_text.empty()
        
        return results

def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica estilos ao DataFrame"""
    def highlight_signals(val):
        if val == "🟢":
            return 'background-color: #90EE90; font-weight: bold'
        elif val == "🔴":
            return 'background-color: #FFB6C1; font-weight: bold'
        elif isinstance(val, str) and '%' in val:
            if val.startswith('+'):
                return 'color: green; font-weight: bold'
            elif val.startswith('-'):
                return 'color: red; font-weight: bold'
        return ''
    
    def highlight_mean_reversion(val):
        if "🔥💚" in val:  # Extreme buy - fundo perfeito
            return 'background-color: #00FF00; color: black; font-weight: bold; font-size: 14px'
        elif "🔥❤️" in val:  # Extreme sell - topo perfeito  
            return 'background-color: #FF0000; color: white; font-weight: bold; font-size: 14px'
        elif "🟢📈" in val:  # Oversold opportunity
            return 'background-color: #90EE90; color: darkgreen; font-weight: bold'
        elif "🔴📉" in val:  # Overbought opportunity
            return 'background-color: #FFB6C1; color: darkred; font-weight: bold'
        elif "🔵⚡" in val:  # Breakout zone
            return 'background-color: #87CEEB; color: darkblue; font-weight: bold'
        elif "⚪" in val:  # Neutral zone
            return 'background-color: #F5F5F5; color: gray; font-weight: normal'
        return ''
    
    def highlight_rsi(val):
        if 'Bull' in val:
            return 'color: green; font-weight: bold'
        elif 'Bear' in val:
            return 'color: red; font-weight: bold'
        return ''
    
    def highlight_stoch(val):
        if "⚡🚀" in val or "💀💥" in val:  # Setups perfeitos
            return 'background-color: #FFD700; color: black; font-weight: bold; font-size: 14px'
        elif "🟢" in val:  # Zona sobrevenda
            return 'background-color: #E8F5E8; color: green; font-weight: bold'
        elif "🔴" in val:  # Zona sobrecompra
            return 'background-color: #FFE8E8; color: red; font-weight: bold'
        elif "🚀" in val:  # Cruzamento alta zona neutra
            return 'background-color: #E8F8FF; color: blue; font-weight: bold'
        elif "💥" in val:  # Cruzamento baixa zona neutra
            return 'background-color: #FFF0E8; color: orange; font-weight: bold'
        return ''
    
    styled = df.style.map(highlight_signals, subset=['B1', 'B2', 'B3'])
    styled = styled.map(highlight_rsi, subset=['RSI'])
    styled = styled.map(highlight_stoch, subset=['Stoch Direção'])
    styled = styled.map(highlight_mean_reversion, subset=['Mean Reversion'])
    
    return styled

def main():
    st.title("₿ Analisador Simplificado - Barreiras EMA")
    st.markdown("**Visualização rápida de todas as criptomoedas com barreiras EMA e indicadores**")
    
    # Mostrar horário atual do Brasil
    brazil_tz = pytz.timezone('America/Sao_Paulo')
    current_time = datetime.now(brazil_tz)
    st.markdown(f"🕐 **Horário BR:** {current_time.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Inicializar analisador
    analyzer = SimplifiedCryptoAnalyzer()
    
    # Configurações em linha
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        timeframe = st.selectbox(
            "Timeframe:",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=6
        )
    
    with col2:
        days = st.number_input(
            "Dias de Histórico:",
            min_value=50,
            max_value=300,
            value=100
        )
    
    with col3:
        scan_button = st.button("🔍 Escanear Todas as Moedas", type="primary", use_container_width=True)
    
    # Legenda
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**RSI 75:**")
        st.markdown("• Bull: > 50")
        st.markdown("• Bear: < 50")
    
    with col2:
        st.markdown("**Barreiras EMA:**")
        st.markdown("• 🟢 Cruzou p/ cima")
        st.markdown("• 🔴 Cruzou p/ baixo")
        st.markdown("• % Distância da linha")
    
    with col3:
        st.markdown("**Estocástico Direção:**")
        st.markdown("• ⚡🚀 Setup PERFEITO compra")
        st.markdown("• 💀💥 Setup PERFEITO venda")
        st.markdown("• 🟢📈📉 Sobrevenda (↑↓)")
        st.markdown("• 🔴📈📉 Sobrecompra (↑↓)")
        st.markdown("• 🚀💥 Cruzamentos zona neutra")

    with col4:
        st.markdown("**Configurações EMA:**")
        st.markdown("• B1: EMA Rápida")
        st.markdown("• B2: EMA Média")
        st.markdown("• B3: EMA Lenta")
    
    with col5:
        st.markdown("**Mean Reversion:**")
        st.markdown("• 🔥💚 EXTREME BUY (fundo)")
        st.markdown("• 🔥❤️ EXTREME SELL (topo)")
        st.markdown("• 🟢📈 Oversold (oportunidade)")
        st.markdown("• 🔴📉 Overbought (oportunidade)")
        st.markdown("• 🔵⚡ Breakout Zone (usar EMAs)")
        st.markdown("• ⚪ Neutro (↗️↘️ direção)")
    
    st.markdown("---")
    
    if scan_button:
        with st.spinner("🔍 Escaneando todas as criptomoedas..."):
            # Executar scanner
            results = analyzer.scan_all_cryptos(timeframe, days)
            
            if results:
                # Converter para DataFrame
                df = pd.DataFrame(results)
                
                # Renomear colunas
                df_display = df[['symbol', 'price', 'rsi', 'b1', 'b2', 'b3', 'stochastic', 'mean_reversion', 'timestamp']].copy()
                df_display.columns = ['Moeda', 'Preço', 'RSI', 'B1', 'B2', 'B3', 'Stoch Direção', 'Mean Reversion', 'Última Atualização']
                
                # Formatar preço
                df_display['Preço'] = df_display['Preço'].apply(lambda x: f"${x:.6f}")
                
                # Aplicar estilos e exibir
                styled_df = style_dataframe(df_display)
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                # Estatísticas resumidas
                st.markdown("---")
                st.subheader("📊 Resumo dos Sinais")
                
                # Contar sinais
                b1_buy = len([r for r in results if r['b1'] == "🟢"])
                b1_sell = len([r for r in results if r['b1'] == "🔴"])
                b2_buy = len([r for r in results if r['b2'] == "🟢"])
                b2_sell = len([r for r in results if r['b2'] == "🔴"])
                b3_buy = len([r for r in results if r['b3'] == "🟢"])
                b3_sell = len([r for r in results if r['b3'] == "🔴"])
                
                # RSI Bull vs Bear
                rsi_bull = len([r for r in results if 'Bull' in r['rsi']])
                rsi_bear = len([r for r in results if 'Bear' in r['rsi']])
                
                # Estocástico
                try:
                    stoch_perfect_buy = len([r for r in results if "⚡🚀" in r['stochastic']])
                    stoch_perfect_sell = len([r for r in results if "💀💥" in r['stochastic']])
                    stoch_oversold = len([r for r in results if "🟢" in r['stochastic']])
                    stoch_overbought = len([r for r in results if "🔴" in r['stochastic']])
                except:
                    stoch_perfect_buy = stoch_perfect_sell = stoch_oversold = stoch_overbought = 0
                
                # Mean Reversion stats
                try:
                    extreme_buy = len([r for r in results if "🔥💚" in r['mean_reversion']])
                    extreme_sell = len([r for r in results if "🔥❤️" in r['mean_reversion']])
                    oversold_ops = len([r for r in results if "🟢📈" in r['mean_reversion']])
                    overbought_ops = len([r for r in results if "🔴📉" in r['mean_reversion']])
                    breakout_zone = len([r for r in results if "🔵⚡" in r['mean_reversion']])
                except:
                    extreme_buy = extreme_sell = oversold_ops = overbought_ops = breakout_zone = 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Analisadas", len(results))
                    st.metric("RSI Bull", rsi_bull, delta=f"{(rsi_bull/len(results)*100):.1f}%")
                    st.metric("RSI Bear", rsi_bear, delta=f"{(rsi_bear/len(results)*100):.1f}%")
                
                with col2:
                    st.metric("B1 Compra 🟢", b1_buy)
                    st.metric("B2 Compra 🟢", b2_buy)
                    st.metric("B3 Compra 🟢", b3_buy)
                
                with col3:
                    st.metric("B1 Venda 🔴", b1_sell)
                    st.metric("B2 Venda 🔴", b2_sell)
                    st.metric("B3 Venda 🔴", b3_sell)
                
                with col4:
                    st.metric("⚡ Setups Perfeitos", stoch_perfect_buy + stoch_perfect_sell)
                    st.metric("⚡🚀 Perfect BUY", stoch_perfect_buy)
                    st.metric("💀💥 Perfect SELL", stoch_perfect_sell)
                    st.metric("🟢 Zona Sobrevenda", stoch_oversold)
                    st.metric("🔴 Zona Sobrecompra", stoch_overbought)
                
                with col5:
                    st.metric("🔥 Extremes Total", extreme_buy + extreme_sell)
                    st.metric("🔥💚 Extreme BUY", extreme_buy)
                    st.metric("🔥❤️ Extreme SELL", extreme_sell)
                    st.metric("🟢 Oversold Ops", oversold_ops)
                    st.metric("🔴 Overbought Ops", overbought_ops)
                
                # Botão para exportar
                st.markdown("---")
                csv_data = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar Dados (CSV)",
                    data=csv_data,
                    file_name=f"crypto_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("❌ Nenhum dado foi obtido. Verifique sua conexão com a internet.")
    
    # Informações adicionais no final
    st.markdown("---")
    st.info("""
**ℹ️ Como interpretar:**

**Barreiras EMA:**
🟢 **Compra**: Preço cruzou EMA de baixo para cima (momentum de alta)
🔴 **Venda**: Preço cruzou EMA de cima para baixo (momentum de baixa)
**% Positiva**: Preço acima da EMA (distância em %)
**% Negativa**: Preço abaixo da EMA (distância em %)

**Stochastic Direção:**
⚡🚀 **Setup PERFEITO Compra**: StochRSI cruzou na sobrevenda
💀💥 **Setup PERFEITO Venda**: StochRSI cruzou na sobrecompra  
🟢📈📉 **Sobrevenda**: <20, subindo/descendo (oportunidade)
🔴📈📉 **Sobrecompra**: >80, subindo/descendo (cuidado)

**Mean Reversion:**
🔥💚 **EXTREME BUY**: Fundo matemático perfeito (Z-Score < -2.0 + RSI < 25)
🔥❤️ **EXTREME SELL**: Topo matemático perfeito (Z-Score > 2.0 + RSI > 75)
🟢📈 **Oversold**: Oportunidade de compra (distância > -10% da média)
🔴📉 **Overbought**: Oportunidade de venda (distância > +10% da média)
🔵⚡ **Breakout Zone**: Use estratégia EMA (próximo da média ±5%)

**Estratégias Combinadas:**
• **🔥 Extremes** = Mean Reversion (maior probabilidade)
• **🔵 Breakout Zone** = Use barreiras EMA + Stochastic
• **⚡ Setups Perfeitos** = Combine ambas as estratégias!

**Dica MASTER**: Priorize 🔥💚 e 🔥❤️ (extremos) > ⚡🚀 e 💀💥 (stoch perfeitos) > demais sinais!
""")
    
    st.markdown("**⚠️ Aviso:** Este não é um conselho financeiro. Sempre faça sua própria análise.")

if __name__ == "__main__":
    main()
