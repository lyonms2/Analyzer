# analysis_tab.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show_analysis_page():
    st.title("📊 Análise de Resultados de Trades")
    st.markdown("Visualize o desempenho das suas operações com base no histórico de trades.")

    # Upload opcional ou leitura direta do arquivo
    uploaded_file = st.file_uploader("📁 Envie um arquivo CSV de histórico de trades", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("trade_history.csv")
        except FileNotFoundError:
            st.warning("⚠️ Nenhum arquivo 'trade_history.csv' encontrado no diretório.")
            st.stop()

    # Verificar colunas
    if "closedPnl" not in df.columns:
        st.error("❌ Coluna 'closedPnl' não encontrada no arquivo.")
        st.stop()

    # Converter para numérico e limpar
    df["closedPnl"] = pd.to_numeric(df["closedPnl"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce")

    # Calcular estatísticas
    total_trades = len(df)
    profitable_trades = (df["closedPnl"] > 0).sum()
    losing_trades = (df["closedPnl"] < 0).sum()
    win_rate = (profitable_trades / (profitable_trades + losing_trades) * 100) if total_trades > 0 else 0
    avg_profit = df.loc[df["closedPnl"] > 0, "closedPnl"].mean()
    avg_loss = df.loc[df["closedPnl"] < 0, "closedPnl"].mean()
    total_pnl = df["closedPnl"].sum()

    # Mostrar métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Trades", total_trades)
    col2.metric("Trades Vencedores", profitable_trades)
    col3.metric("Trades Perdidos", losing_trades)
    col4.metric("Taxa de Acerto", f"{win_rate:.2f}%")
    col5.metric("PnL Total", f"{total_pnl:.2f}")

    # Gráfico de PnL acumulado
    df["PnL_Acumulado"] = df["closedPnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["PnL_Acumulado"],
        mode="lines+markers",
        name="PnL Acumulado",
        line=dict(color="cyan", width=3)
    ))
    fig.update_layout(
        title="💰 Evolução do PnL Acumulado",
        xaxis_title="Trade #",
        yaxis_title="PnL (USDT)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar tabela completa
    with st.expander("📋 Ver tabela completa de trades"):
        st.dataframe(df, use_container_width=True, height=600)

    # Estatísticas detalhadas
    st.markdown("### 📈 Estatísticas Detalhadas")
    stats = {
        "Lucro Médio": avg_profit,
        "Perda Média": avg_loss,
        "Maior Lucro": df["closedPnl"].max(),
        "Maior Perda": df["closedPnl"].min(),
        "Taxas Pagas (Total)": df["fee"].sum(),
    }
    st.table(pd.DataFrame(stats, index=["Valor (USDT)"]))

