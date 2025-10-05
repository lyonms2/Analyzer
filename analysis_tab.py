# analysis_tab.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def show_analysis_page():
    st.title("📊 Análise de Resultados de Trades")
    st.markdown("Análise baseada apenas nos trades **fechados** (`Close Long` e `Close Short`), com PnL bruto e líquido calculados.")

    # Upload opcional
    uploaded_file = st.file_uploader("📁 Envie um arquivo CSV de histórico de trades", type=["csv"])
    if uploaded_file is not None:
        # Inferir delimitador e aceitar decimal com vírgula
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        try:
            df = pd.read_csv("trade_history.csv", sep=None, engine="python")
        except FileNotFoundError:
            st.warning("⚠️ Nenhum arquivo 'trade_history.csv' encontrado no diretório.")
            st.stop()

    # Conversões
    # Normalizar decimais com vírgula e converter para número
    if "closedPnl" in df.columns:
        df["closedPnl"] = pd.to_numeric(df["closedPnl"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if "fee" in df.columns:
        df["fee"] = pd.to_numeric(df["fee"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%d/%m/%Y - %H:%M:%S")

    # Separar aberturas e fechamentos
    df_open = df[df["dir"].isin(["Open Long", "Open Short"])].copy()
    df_close = df[df["dir"].isin(["Close Long", "Close Short"])].copy()

    if df_close.empty:
        st.error("❌ Nenhum trade fechado encontrado (Close Long / Close Short).")
        st.stop()

    # Vincular taxa do Open mais próximo anterior ao Close da mesma coin e direção (opcional)
    df_close["open_fee"] = 0.0
    for idx, row in df_close.iterrows():
        same_coin = df_open[df_open["coin"] == row["coin"]]
        if "Long" in row["dir"]:
            same_coin = same_coin[same_coin["dir"] == "Open Long"]
        else:
            same_coin = same_coin[same_coin["dir"] == "Open Short"]

        same_coin = same_coin[same_coin["time"] <= row["time"]]
        if not same_coin.empty:
            last_open = same_coin.iloc[-1]
            df_close.at[idx, "open_fee"] = last_open["fee"]

    # Calcular PnL: considerar apenas closedPnl conforme solicitado
    df_close["pnl_bruto"] = df_close["closedPnl"].fillna(0)
    df_close["pnl_liquido"] = df_close["closedPnl"].fillna(0)

    # Classificar resultado (com base em closedPnl)
    df_close["resultado"] = df_close["closedPnl"].apply(lambda x: "Acerto" if x > 0 else ("Erro" if x < 0 else "Neutro"))

    # Estatísticas
    total_trades = len(df_close)
    acertos = (df_close["resultado"] == "Acerto").sum()
    erros = (df_close["resultado"] == "Erro").sum()
    taxa_acerto = (acertos / (acertos + erros) * 100) if (acertos + erros) > 0 else 0

    # Totais solicitados
    total_closedpnl_arquivo = df["closedPnl"].sum()
    total_closedpnl_fechados = df_close["closedPnl"].sum()

    lucro_medio = df_close.loc[df_close["closedPnl"] > 0, "closedPnl"].mean()
    perda_media = df_close.loc[df_close["closedPnl"] < 0, "closedPnl"].mean()

    # Métricas principais
    st.markdown("### 💹 Resultados Gerais")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total de Trades Fechados", total_trades)
    col2.metric("Acertos", acertos)
    col3.metric("Erros", erros)
    col4.metric("Taxa de Acerto", f"{taxa_acerto:.2f}%")
    col5.metric("Total closedPnl (arquivo inteiro)", f"{total_closedpnl_arquivo:.4f}")
    col6.metric("Total closedPnl (apenas Close)", f"{total_closedpnl_fechados:.4f}")

    # PnL acumulado (líquido e bruto)
    df_close["PnL_Acumulado_Liquido"] = df_close["closedPnl"].cumsum()
    df_close["PnL_Acumulado_Bruto"] = df_close["pnl_bruto"].cumsum()

    fig_pnl = go.Figure()
    # Linha principal: Líquido
    fig_pnl.add_trace(go.Scatter(
        x=df_close["time"],
        y=df_close["PnL_Acumulado_Liquido"],
        mode="lines+markers",
        name="PnL Líquido (Acumulado)",
        line=dict(color="cyan", width=3)
    ))
    # Linha secundária: Bruto
    fig_pnl.add_trace(go.Scatter(
        x=df_close["time"],
        y=df_close["PnL_Acumulado_Bruto"],
        mode="lines",
        name="PnL Bruto (Acumulado)",
        line=dict(color="lightgray", width=2, dash="dash")
    ))

    fig_pnl.update_layout(
        title="💰 Evolução do PnL Acumulado",
        xaxis_title="Data/Hora",
        yaxis_title="PnL (USDT)",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ========================
    # 🪙 Desempenho por Moeda
    # ========================
    st.markdown("### 🪙 Desempenho por Moeda")
    coin_stats = (
        df_close.groupby("coin")
        .agg(
            total_trades=("closedPnl", "count"),
            acertos=("closedPnl", lambda x: (x > 0).sum()),
            erros=("closedPnl", lambda x: (x < 0).sum()),
            taxa_acerto=("closedPnl", lambda x: (x > 0).sum() / len(x) * 100),
            pnl_bruto=("closedPnl", "sum"),
            pnl_liquido=("closedPnl", "sum")
        )
        .sort_values("taxa_acerto", ascending=False)
        .reset_index()
    )
    st.dataframe(coin_stats, use_container_width=True)

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
    # ⏰ Desempenho por Hora
    # ========================
    st.markdown("### ⏰ Desempenho por Hora do Dia")
    df_close["hora"] = df_close["time"].dt.hour
    hora_stats = (
        df_close.groupby("hora")
        .agg(
            total_trades=("closedPnl", "count"),
            acertos=("closedPnl", lambda x: (x > 0).sum()),
            erros=("closedPnl", lambda x: (x < 0).sum()),
            taxa_acerto=("closedPnl", lambda x: (x > 0).sum() / len(x) * 100),
            pnl_bruto=("closedPnl", "sum"),
            pnl_liquido=("closedPnl", "sum")
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
    # 📋 Tabela Detalhada
    # ========================
    with st.expander("📋 Ver tabela completa de trades fechados"):
        st.dataframe(df_close, use_container_width=True, height=600)
