# analysis_tab.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def show_analysis_page():
    st.title("📊 Análise de Resultados de Trades")
    st.markdown("Análise de desempenho com base nos trades **fechados** (`Close Long` ou `Close Short`).")

    # Upload opcional ou leitura local
    uploaded_file = st.file_uploader("📁 Envie um arquivo CSV de histórico de trades", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("trade_history.csv")
        except FileNotFoundError:
            st.warning("⚠️ Nenhum arquivo 'trade_history.csv' encontrado no diretório.")
            st.stop()

    # Converter tipos
    df["closedPnl"] = pd.to_numeric(df["closedPnl"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%d/%m/%Y - %H:%M:%S")

    # Filtrar apenas trades fechados
    df_trades = df[df["dir"].isin(["Close Long", "Close Short"])].copy()

    if df_trades.empty:
        st.error("❌ Nenhum trade fechado encontrado (Close Long / Close Short).")
        st.stop()

    # Classificação de acertos/erros
    df_trades["resultado"] = df_trades["closedPnl"].apply(lambda x: "Acerto" if x > 0 else ("Erro" if x < 0 else "Neutro"))

    # Estatísticas principais
    total_trades = len(df_trades)
    acertos = (df_trades["resultado"] == "Acerto").sum()
    erros = (df_trades["resultado"] == "Erro").sum()
    taxa_acerto = (acertos / (acertos + erros) * 100) if (acertos + erros) > 0 else 0
    pnl_total = df_trades["closedPnl"].sum()
    lucro_medio = df_trades.loc[df_trades["closedPnl"] > 0, "closedPnl"].mean()
    perda_media = df_trades.loc[df_trades["closedPnl"] < 0, "closedPnl"].mean()

    # Métricas principais
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Trades Fechados", total_trades)
    col2.metric("Acertos", acertos)
    col3.metric("Erros", erros)
    col4.metric("Taxa de Acerto", f"{taxa_acerto:.2f}%")
    col5.metric("PnL Total", f"{pnl_total:.2f}")

    # PnL acumulado
    df_trades["PnL_Acumulado"] = df_trades["closedPnl"].cumsum()
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=df_trades["time"],
        y=df_trades["PnL_Acumulado"],
        mode="lines+markers",
        name="PnL Acumulado",
        line=dict(color="cyan", width=3)
    ))
    fig_pnl.update_layout(
        title="💰 Evolução do PnL Acumulado (Apenas Trades Fechados)",
        xaxis_title="Data/Hora",
        yaxis_title="PnL (USDT)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ========================
    # 🔍 Análise por moeda
    # ========================
    st.markdown("### 🪙 Desempenho por Moeda")
    coin_stats = (
        df_trades.groupby("coin")
        .agg(
            total_trades=("closedPnl", "count"),
            acertos=("closedPnl", lambda x: (x > 0).sum()),
            erros=("closedPnl", lambda x: (x < 0).sum()),
            taxa_acerto=("closedPnl", lambda x: (x > 0).sum() / len(x) * 100),
            pnl_total=("closedPnl", "sum")
        )
        .sort_values("taxa_acerto", ascending=False)
        .reset_index()
    )

    st.dataframe(coin_stats, use_container_width=True)

    # Gráfico de taxa de acerto por moeda
    fig_coin = px.bar(
        coin_stats,
        x="coin",
        y="taxa_acerto",
        text_auto=".2f",
        title="🎯 Taxa de Acerto por Moeda",
        color="taxa_acerto",
        color_continuous_scale="Tealgrn"
    )
    fig_coin.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_coin, use_container_width=True)

    # ========================
    # 🕒 Análise por hora do dia
    # ========================
    st.markdown("### ⏰ Desempenho por Hora do Dia")
    df_trades["hora"] = df_trades["time"].dt.hour

    hora_stats = (
        df_trades.groupby("hora")
        .agg(
            total_trades=("closedPnl", "count"),
            acertos=("closedPnl", lambda x: (x > 0).sum()),
            erros=("closedPnl", lambda x: (x < 0).sum()),
            taxa_acerto=("closedPnl", lambda x: (x > 0).sum() / len(x) * 100),
            pnl_total=("closedPnl", "sum")
        )
        .reset_index()
    )

    fig_hora = px.bar(
        hora_stats,
        x="hora",
        y="taxa_acerto",
        text_auto=".1f",
        title="🕒 Taxa de Acerto por Hora do Dia",
        color="taxa_acerto",
        color_continuous_scale="Viridis"
    )
    fig_hora.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_hora, use_container_width=True)

    # ========================
    # 📋 Tabela detalhada
    # ========================
    with st.expander("📋 Ver tabela completa de trades fechados"):
        st.dataframe(df_trades, use_container_width=True, height=600)
