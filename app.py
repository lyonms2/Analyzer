import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
from typing import Dict, List
import time
import pytz
from datetime import datetime
from ema_barriers_config import CRYPTO_LIST

# Configuração da página
st.set_page_config(
    page_title="Analisador de Sinais - Crypto",
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
            # Preço atual
            current_close = data['Close'].iloc[-1]
            
            # RSI 75 períodos
            rsi_75 = ta.momentum.RSIIndicator(close=data['Close'], window=75).rsi().iloc[-1]
            
            # Determinar tendência RSI
            if rsi_75 > 50:
                rsi_signal = f"{rsi_75:05.1f} 🟢 Bull"
            else:
                rsi_signal = f"{rsi_75:05.1f} 🔴 Bear"
            
            # Estocástico RSI
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
            
            # Detectar cruzamentos e zonas
            def analyze_stochastic(k_curr, k_prev, d_curr, d_prev):
                # Detectar cruzamentos
                k_cross_d_up = k_prev <= d_prev and k_curr > d_curr
                k_cross_d_down = k_prev >= d_prev and k_curr < d_curr
                
                # Determinar zona
                in_oversold = k_curr < 20
                in_overbought = k_curr > 80
                
                # Sinais de trading
                if in_overbought and k_cross_d_down:
                    return f"{k_curr:05.1f} 🔴💥 VENDA FORTE"  # Cruzamento em sobrecompra = VENDA
                elif in_oversold and k_cross_d_up:
                    return f"{k_curr:05.1f} 🟢🚀 COMPRA FORTE"  # Cruzamento em sobrevenda = COMPRA
                elif in_overbought:
                    return f"{k_curr:05.1f} 🔴 Sobrecomprado"  # Apenas sobrecompra
                elif in_oversold:
                    return f"{k_curr:05.1f} 🟢 Sobrevendido"  # Apenas sobrevenda
                else:
                    return f"{k_curr:05.1f} ⚪ Neutro"  # Zona neutra
            
            stoch_signal = analyze_stochastic(k_current, k_previous, d_current, d_previous)
            
            # Mean Reversion baseado em EMA 50
            ema_50 = ta.trend.EMAIndicator(close=data['Close'], window=50).ema_indicator().iloc[-1]
            
            # Calcular distância da EMA 50
            distance_pct = ((current_close - ema_50) / ema_50) * 100
            
            # Calcular desvio padrão para determinar extremos
            std_50 = data['Close'].rolling(window=50).std().iloc[-1]
            z_score = (current_close - ema_50) / std_50
            
            # RSI 14 para confluência
            rsi_14_current = rsi_14.iloc[-1]
            
            def analyze_mean_reversion(distance_pct, z_score, rsi_14):
                """Analisa oportunidades de mean reversion baseado na EMA 50"""
                
                # Extremos fortes (melhor probabilidade)
                extreme_oversold = z_score < -2.0 and rsi_14 < 30
                extreme_overbought = z_score > 2.0 and rsi_14 > 70
                
                # Oportunidades de mean reversion
                oversold_opportunity = z_score < -1.5 and distance_pct < -8
                overbought_opportunity = z_score > 1.5 and distance_pct > 8
                
                # Próximo da média (não operar mean reversion)
                near_mean = abs(z_score) < 0.5 and abs(distance_pct) < 3
                
                # Sinais visuais
                if extreme_oversold:
                    return f"{distance_pct:+06.1f}% 🔥💚 COMPRA EXTREMA"
                elif extreme_overbought:
                    return f"{distance_pct:+06.1f}% 🔥❤️ VENDA EXTREMA"
                elif oversold_opportunity:
                    return f"{distance_pct:+06.1f}% 🟢📈 Oportunidade Compra"
                elif overbought_opportunity:
                    return f"{distance_pct:+06.1f}% 🔴📉 Oportunidade Venda"
                elif near_mean:
                    return f"{distance_pct:+06.1f}% 🔵 Próximo EMA50"
                else:
                    # Zona neutra
                    if distance_pct > 0:
                        return f"{distance_pct:+06.1f}% ⚪ Acima EMA50"
                    else:
                        return f"{distance_pct:+06.1f}% ⚪ Abaixo EMA50"
            
            mean_reversion_signal = analyze_mean_reversion(distance_pct, z_score, rsi_14_current)
            
            return {
                'symbol': symbol,
                'price': current_close,
                'rsi': rsi_signal,
                'stochastic': stoch_signal,
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
    
    def highlight_rsi(val):
        if '🟢 Bull' in val:
            return 'background-color: #E8F5E8; color: darkgreen; font-weight: bold'
        elif '🔴 Bear' in val:
            return 'background-color: #FFE8E8; color: darkred; font-weight: bold'
        return ''
    
    def highlight_stoch(val):
        if "🔴💥 VENDA FORTE" in val:
            return 'background-color: #FF0000; color: white; font-weight: bold; font-size: 14px'
        elif "🟢🚀 COMPRA FORTE" in val:
            return 'background-color: #00FF00; color: black; font-weight: bold; font-size: 14px'
        elif "🔴 Sobrecomprado" in val:
            return 'background-color: #FFB6C1; color: darkred; font-weight: bold'
        elif "🟢 Sobrevendido" in val:
            return 'background-color: #90EE90; color: darkgreen; font-weight: bold'
        elif "⚪ Neutro" in val:
            return 'background-color: #F5F5F5; color: gray'
        return ''
    
    def highlight_mean_reversion(val):
        if "🔥💚 COMPRA EXTREMA" in val:
            return 'background-color: #00FF00; color: black; font-weight: bold; font-size: 14px'
        elif "🔥❤️ VENDA EXTREMA" in val:
            return 'background-color: #FF0000; color: white; font-weight: bold; font-size: 14px'
        elif "🟢📈 Oportunidade Compra" in val:
            return 'background-color: #90EE90; color: darkgreen; font-weight: bold'
        elif "🔴📉 Oportunidade Venda" in val:
            return 'background-color: #FFB6C1; color: darkred; font-weight: bold'
        elif "🔵 Próximo EMA50" in val:
            return 'background-color: #ADD8E6; color: darkblue; font-weight: bold'
        elif "⚪" in val:
            return 'background-color: #F5F5F5; color: gray'
        return ''
    
    styled = df.style.map(highlight_rsi, subset=['RSI 75'])
    styled = styled.map(highlight_stoch, subset=['Estocástico'])
    styled = styled.map(highlight_mean_reversion, subset=['Mean Reversion EMA50'])
    
    return styled

def main():
    st.title("₿ Analisador de Sinais - Crypto")
    st.markdown("**Sinais claros de trading: RSI 75, Estocástico e Mean Reversion (EMA 50)**")
    
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 RSI 75 Períodos:**")
        st.markdown("• 🟢 **Bull**: RSI > 50 (tendência de alta)")
        st.markdown("• 🔴 **Bear**: RSI < 50 (tendência de baixa)")
    
    with col2:
        st.markdown("**🎯 Estocástico RSI:**")
        st.markdown("• 🟢🚀 **COMPRA FORTE**: Cruzamento em sobrevenda")
        st.markdown("• 🔴💥 **VENDA FORTE**: Cruzamento em sobrecompra")
        st.markdown("• 🟢 **Sobrevendido**: StochRSI < 20")
        st.markdown("• 🔴 **Sobrecomprado**: StochRSI > 80")
        st.markdown("• ⚪ **Neutro**: Entre 20-80")
    
    with col3:
        st.markdown("**📈 Mean Reversion (EMA 50):**")
        st.markdown("• 🔥💚 **COMPRA EXTREMA**: Fundo extremo (Z<-2 + RSI<30)")
        st.markdown("• 🔥❤️ **VENDA EXTREMA**: Topo extremo (Z>2 + RSI>70)")
        st.markdown("• 🟢📈 **Oportunidade Compra**: Distância > -8%")
        st.markdown("• 🔴📉 **Oportunidade Venda**: Distância > +8%")
        st.markdown("• 🔵 **Próximo EMA50**: ±3% da média")
    
    st.markdown("---")
    
    if scan_button:
        with st.spinner("🔍 Escaneando todas as criptomoedas..."):
            # Executar scanner
            results = analyzer.scan_all_cryptos(timeframe, days)
            
            if results:
                # Converter para DataFrame
                df = pd.DataFrame(results)
                
                # Renomear colunas
                df_display = df[['symbol', 'price', 'rsi', 'stochastic', 'mean_reversion', 'timestamp']].copy()
                df_display.columns = ['Moeda', 'Preço', 'RSI 75', 'Estocástico', 'Mean Reversion EMA50', 'Última Atualização']
                
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
                rsi_bull = len([r for r in results if '🟢 Bull' in r['rsi']])
                rsi_bear = len([r for r in results if '🔴 Bear' in r['rsi']])
                
                stoch_buy_strong = len([r for r in results if "🟢🚀 COMPRA FORTE" in r['stochastic']])
                stoch_sell_strong = len([r for r in results if "🔴💥 VENDA FORTE" in r['stochastic']])
                stoch_oversold = len([r for r in results if "🟢 Sobrevendido" in r['stochastic']])
                stoch_overbought = len([r for r in results if "🔴 Sobrecomprado" in r['stochastic']])
                
                mean_extreme_buy = len([r for r in results if "🔥💚 COMPRA EXTREMA" in r['mean_reversion']])
                mean_extreme_sell = len([r for r in results if "🔥❤️ VENDA EXTREMA" in r['mean_reversion']])
                mean_buy_opp = len([r for r in results if "🟢📈 Oportunidade Compra" in r['mean_reversion']])
                mean_sell_opp = len([r for r in results if "🔴📉 Oportunidade Venda" in r['mean_reversion']])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Analisadas", len(results))
                    st.metric("🟢 RSI Bull", rsi_bull, delta=f"{(rsi_bull/len(results)*100):.1f}%")
                    st.metric("🔴 RSI Bear", rsi_bear, delta=f"{(rsi_bear/len(results)*100):.1f}%")
                
                with col2:
                    st.metric("🎯 Sinais de Estocástico", stoch_buy_strong + stoch_sell_strong)
                    st.metric("🟢🚀 COMPRA FORTE", stoch_buy_strong)
                    st.metric("🔴💥 VENDA FORTE", stoch_sell_strong)
                    st.metric("🟢 Sobrevendido", stoch_oversold)
                    st.metric("🔴 Sobrecomprado", stoch_overbought)
                
                with col3:
                    st.metric("🔥 Extremos Total", mean_extreme_buy + mean_extreme_sell)
                    st.metric("🔥💚 COMPRA EXTREMA", mean_extreme_buy)
                    st.metric("🔥❤️ VENDA EXTREMA", mean_extreme_sell)
                    st.metric("🟢 Oportunidades Compra", mean_buy_opp)
                    st.metric("🔴 Oportunidades Venda", mean_sell_opp)
                
                # Botão para exportar
                st.markdown("---")
                csv_data = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar Dados (CSV)",
                    data=csv_data,
                    file_name=f"crypto_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("❌ Nenhum dado foi obtido. Verifique sua conexão com a internet.")
    
    # Informações adicionais no final
    st.markdown("---")
    st.info("""
**🎯 GUIA RÁPIDO DE INTERPRETAÇÃO:**

**1️⃣ RSI 75 (Tendência Geral):**
• 🟢 Bull (>50): Mercado em tendência de alta
• 🔴 Bear (<50): Mercado em tendência de baixa

**2️⃣ Estocástico RSI (Timing de Entrada):**
• 🟢🚀 **COMPRA FORTE**: Melhor momento para compra (cruzou em sobrevenda)
• 🔴💥 **VENDA FORTE**: Melhor momento para venda (cruzou em sobrecompra)
• 🟢 Sobrevendido: Atenção para possível reversão de alta
• 🔴 Sobrecomprado: Atenção para possível reversão de baixa

**3️⃣ Mean Reversion EMA 50 (Estratégia de Retorno à Média):**
• 🔥💚 **COMPRA EXTREMA**: Maior probabilidade de retorno (usar esta estratégia)
• 🔥❤️ **VENDA EXTREMA**: Maior probabilidade de retorno (usar esta estratégia)
• 🟢📈 Oportunidade Compra: Preço muito abaixo da EMA 50
• 🔴📉 Oportunidade Venda: Preço muito acima da EMA 50
• 🔵 Próximo EMA50: Aguardar melhor momento

**💡 ESTRATÉGIA RECOMENDADA:**

**Para COMPRAS:**
1. Priorize: 🔥💚 COMPRA EXTREMA (melhor probabilidade)
2. Confirme com: 🟢🚀 COMPRA FORTE no estocástico
3. Verifique: 🟢 Bull no RSI 75 para confluência

**Para VENDAS:**
1. Priorize: 🔥❤️ VENDA EXTREMA (melhor probabilidade)
2. Confirme com: 🔴💥 VENDA FORTE no estocástico
3. Verifique: 🔴 Bear no RSI 75 para confluência

**⚠️ IMPORTANTE:**
• Sinais 🔥 (extremos) têm maior probabilidade de acerto
• Sinais 🚀💥 (cruzamentos em zonas extremas) são excelentes para timing
• Combine os indicadores para maior confiança
• Use stop loss sempre!
""")
    
    st.markdown("**⚠️ Aviso Legal:** Este não é um conselho financeiro. Sempre faça sua própria análise antes de operar.")

if __name__ == "__main__":
    main()
