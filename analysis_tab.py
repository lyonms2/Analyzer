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
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("trade_history.csv")
        except FileNotFoundError:
            st.warning("⚠️ Nenhum arquivo 'trade_history.csv' encontrado no diretório.")
            st.stop()

    # Conversões
    df["closedPnl"] = pd.to_numeric(df["closedPnl"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%d/%m/%Y - %H:%M:%S")

    # Separar aberturas e fechamentos
    df_open = df[df["dir"].isin(["Open Long", "Open Short"])].copy()
    df_close = df[df["dir"].isin(["Close Long", "Close Short"])].copy()

    if df_close.empty:
        st.error("❌ Nenhum trade fechado encontrado (Close Long / Close Short).")
        st.stop()

    # Vincular taxa e PnL do Open mais próximo anterior ao Close da mesma coin e direção
    df_close["open_fee"] = 0.0
    df_close["open_pnl"] = 0.0
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
            # incluir PnL do evento de abertura (se existir no CSV)
            if "closedPnl" in last_open:
                df_close.at[idx, "open_pnl"] = last_open["closedPnl"]

    # Calcular taxas totais e PnL (incluindo PnL do Open associado)
    df_close["total_fee"] = df_close["fee"] + df_close["open_fee"]
    # PnL bruto: soma do PnL do Open + Close (sem taxas)
    df_close["pnl_bruto"] = df_close["closedPnl"].fillna(0) + df_close["open_pnl"].fillna(0)
    # PnL líquido: bruto menos taxas de abertura e fechamento
    df_close["pnl_liquido"] = df_close["pnl_bruto"] - df_close["total_fee"]

    # Classificar resultado (com base no PnL líquido)
    df_close["resultado"] = df_close["pnl_liquido"].apply(lambda x: "Acerto" if x > 0 else ("Erro" if x < 0 else "Neutro"))

    # Estatísticas
    total_trades = len(df_close)
    acertos = (df_close["resultado"] == "Acerto").sum()
    erros = (df_close["resultado"] == "Erro").sum()
    taxa_acerto = (acertos / (acertos + erros) * 100) if (acertos + erros) > 0 else 0

    pnl_bruto_total = df_close["pnl_bruto"].sum()
    pnl_liquido_total = df_close["pnl_liquido"].sum()

    lucro_medio = df_close.loc[df_close["pnl_liquido"] > 0, "pnl_liquido"].mean()
    perda_media = df_close.loc[df_close["pnl_liquido"] < 0, "pnl_liquido"].mean()

    # Métricas principais
    st.markdown("### 💹 Resultados Gerais")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total de Trades Fechados", total_trades)
    col2.metric("Acertos", acertos)
    col3.metric("Erros", erros)
    col4.metric("Taxa de Acerto", f"{taxa_acerto:.2f}%")
    col5.metric("PnL Bruto (sem taxas)", f"{pnl_bruto_total:.4f}")
    col6.metric("PnL Líquido (com taxas)", f"{pnl_liquido_total:.4f}")

    # PnL acumulado (líquido)
    df_close["PnL_Acumulado"] = df_close["pnl_liquido"].cumsum()
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=df_close["time"],
        y=df_close["PnL_Acumulado"],
        mode="lines+markers",
        name="PnL Líquido Acumulado",
        line=dict(color="cyan", width=3)
    ))
    fig_pnl.update_layout(
        title="💰 Evolução do PnL Acumulado (Líquido)",
        xaxis_title="Data/Hora",
        yaxis_title="PnL (USDT)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ========================
    # 🪙 Desempenho por Moeda
    # ========================
    st.markdown("### 🪙 Desempenho por Moeda")
    coin_stats = (
        df_close.groupby("coin")
        .agg(
            total_trades=("pnl_liquido", "count"),
            acertos=("pnl_liquido", lambda x: (x > 0).sum()),
            erros=("pnl_liquido", lambda x: (x < 0).sum()),
            taxa_acerto=("pnl_liquido", lambda x: (x > 0).sum() / len(x) * 100),
            pnl_bruto=("pnl_bruto", "sum"),
            pnl_liquido=("pnl_liquido", "sum")
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
            total_trades=("pnl_liquido", "count"),
            acertos=("pnl_liquido", lambda x: (x > 0).sum()),
            erros=("pnl_liquido", lambda x: (x < 0).sum()),
            taxa_acerto=("pnl_liquido", lambda x: (x > 0).sum() / len(x) * 100),
            pnl_bruto=("pnl_bruto", "sum"),
            pnl_liquido=("pnl_liquido", "sum")
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
