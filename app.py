# app.py
# ================================================================
# "Calouro - Pagante" Unificado
# Streamlit + Plotly • Aba 1: DELTAS (5 dispersões)
# Aba 2: Percentuais (5 dispersões + Heatmap Pearson + Spearman + Barras IC95%)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import sqrt
from scipy.stats import pearsonr, norm

st.set_page_config(page_title="Calouro - Pagante • Dashboard", layout="wide")

ROXO = "#9900FF"

# ========= Helpers =========
def to_float(x):
    """Converte '1,2 pp' ou '35,5%' -> float (1.2 / 35.5)."""
    if pd.isna(x) or x == "":
        return np.nan
    s = str(x).lower().replace("%","").replace("pp","").replace(",",".").strip()
    try:
        return float(s)
    except:
        return np.nan

def add_quadrants(fig, x0, x1, y0, y1):
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
    fig.add_vline(x=0, line_dash="dash", opacity=0.6)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])

def scatter(df, x_col, y_col, x_label, y_label, title):
    x = df[x_col]; y = df[y_col]
    dx = (x.max() - x.min())*0.1 if np.isfinite(x.max()-x.min()) else 1
    dy = (y.max() - y.min())*0.1 if np.isfinite(y.max()-y.min()) else 1
    x0, x1 = x.min()-dx, x.max()+dx
    y0, y1 = y.min()-dy, y.max()+dy

    fig = px.scatter(
        df, x=x_col, y=y_col, text="MARCA",
        color="AUM", color_continuous_scale="Viridis_r",
        labels={x_col: x_label, y_col: y_label, "AUM":"Δ % AUM"},
        title=title
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=0.6, color="black")),
                      textposition="top center")
    add_quadrants(fig, x0, x1, y0, y1)
    fig.update_layout(margin=dict(l=10, r=10, t=48, b=10))
    return fig

def heatmap_corr(df, cols, method="pearson", title_suffix=""):
    corr = df[cols].corr(method=method)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Correlação")
    ))
    # anotações por célula
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            fig.add_annotation(
                x=col, y=row, text=f"{corr.iloc[i,j]:.2f}",
                showarrow=False, font=dict(size=12, color="black"),
                xanchor="center", yanchor="middle",
                bgcolor="rgba(255,255,255,0.7)"
            )
    fig.update_layout(
        title=f"Heatmap de Correlação — {method.capitalize()} {title_suffix}".strip(),
        xaxis=dict(tickangle=45),
        margin=dict(l=80, r=20, t=60, b=120),
    )
    return fig

def fisher_ci(r, n, alpha=0.05):
    """IC 95% do r de Pearson (Fisher)."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha/2.0)
    return np.tanh(z - zcrit*se), np.tanh(z + zcrit*se)

def barras_correlacoes(df, x_var, var_map, title):
    labels, rvals, lo_list, hi_list = [], [], [], []
    for col, lbl in var_map.items():
        sub = df[[x_var, col]].dropna()
        n = len(sub)
        if n < 3:
            r, lo, hi = np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(sub[x_var], sub[col])
            lo, hi = fisher_ci(r, n)
        labels.append(lbl); rvals.append(r); lo_list.append(lo); hi_list.append(hi)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rvals,
        marker_color=ROXO, marker_line=dict(color="black", width=0.8),
        name="Correlação (Pearson)"
    ))
    y_err_plus  = [ (hi - r) if np.isfinite(hi) and np.isfinite(r) else 0 for r, hi in zip(rvals, hi_list) ]
    y_err_minus = [ (r - lo) if np.isfinite(lo) and np.isfinite(r) else 0 for r, lo in zip(rvals, lo_list) ]
    fig.update_traces(error_y=dict(
        type='data', symmetric=False, array=y_err_plus, arrayminus=y_err_minus,
        thickness=1.2, width=5, color="black"
    ))
    fig.add_hline(y=0, line_dash="dash", opacity=0.7)
    fig.update_layout(
        title=title,
        yaxis_title="Coeficiente de Correlação (Pearson)",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

# ========= Base padrão: Percentuais (Calouro) =========
raw_perc = [
 ["ÂNIMA BR","71,1%","69,0%","18,3%","30,2%","36,9%","22,4%","112,9%","-37,1 pp"],
 ["AGES","77,6%","74,9%","13,4%","24,7%","20,9%","17,4%","154,9%","-45,7 pp"],
 ["UNIFG - BA","82,8%","78,2%","11,5%","21,6%","24,5%","16,1%","140,5%","-49,8 pp"],
 ["UNF","62,9%","62,8%","23,4%","36,7%","31,9%","25,3%","151,6%","-28,1 pp"],
 ["UNP","69,2%","67,7%","20,3%","31,9%","33,8%","22,2%","170,5%","-33,6 pp"],
 ["FPB","50,1%","60,3%","26,3%","38,7%","66,0%","29,2%","160,0%","-28,1 pp"],
 ["UNIFG - PE","55,6%","62,8%","24,1%","35,9%","53,3%","25,9%","167,5%","-30,3 pp"],
 ["UAM","74,1%","70,8%","17,1%","28,2%","41,4%","20,4%","111,7%","0,0 pp"],
 ["USJT","75,3%","69,1%","17,1%","30,0%","41,5%","20,7%","58,1%","0,0 pp"],
 ["UNA","75,7%","71,8%","16,3%","27,3%","36,4%","21,5%","70,3%","0,0 pp"],
 ["UNIBH","73,5%","73,1%","14,1%","26,5%","28,5%","20,9%","109,6%","0,0 pp"],
 ["IBMR","59,6%","66,1%","19,9%","32,7%","40,9%","21,7%","231,0%","-49,5 pp"],
 ["FASEH","75,3%","73,5%","16,0%","25,5%","41,1%","24,4%","63,2%","0,0 pp"],
 ["MIL. CAMPOS","88,9%","76,9%","9,4%","23,1%","0,0%","17,1%","44,9%","0,0 pp"],
 ["UNISUL","67,6%","67,0%","20,9%","32,6%","33,1%","26,0%","80,0%","0,0 pp"],
 ["UNICURITIBA","75,3%","70,4%","16,8%","29,0%","35,1%","19,3%","74,4%","0,0 pp"],
 ["UNISOCIESC","69,7%","68,4%","19,7%","30,8%","39,4%","31,8%","73,1%","0,0 pp"],
 ["UNR","70,8%","68,9%","18,3%","30,2%","34,1%","22,0%","149,6%","-47,5 pp"],
 ["FAD","60,2%","65,1%","22,5%","34,5%","49,1%","29,3%","168,0%","-26,5 pp"]
]
cols_perc = ["MARCA","P_CONV","P_REND100","P_MENOR40","P_REPROV","P_NAO_AI","P_MIX_INAD","AUM","D_DEV_BOLSA"]
df_perc = pd.DataFrame(raw_perc, columns=cols_perc)
for c in cols_perc[1:]:
    df_perc[c] = df_perc[c].apply(to_float)

NUM_COLS_PERC = ["P_CONV","P_REND100","P_MENOR40","P_REPROV","P_NAO_AI","P_MIX_INAD","AUM","D_DEV_BOLSA"]

# ========= Base padrão: DELTAS (Calouro) =========
raw_deltas = [
 ["ÂNIMA BR","-1,2 pp","-3,9 pp","1,7 pp","3,3 pp","-24,8 pp","-3,1 pp","36,1%","-37,1 pp"],
 ["AGES","-2,0 pp","-5,1 pp","1,8 pp","5,8 pp","-36,1 pp","-2,5 pp","102,9%","-45,7 pp"],
 ["UNIFG - BA","1,5 pp","-3,7 pp","1,1 pp","3,4 pp","-17,8 pp","-3,1 pp","126,9%","-49,8 pp"],
 ["UNF","-5,5 pp","-3,2 pp","1,7 pp","2,8 pp","-24,6 pp","-3,6 pp","88,4%","-28,1 pp"],
 ["UNP","-2,1 pp","-4,5 pp","2,7 pp","4,2 pp","-26,9 pp","-3,8 pp","88,3%","-33,6 pp"],
 ["FPB","-8,4 pp","1,7 pp","0,0 pp","-2,6 pp","-0,2 pp","-6,7 pp","95,6%","-28,1 pp"],
 ["UNIFG - PE","-4,5 pp","-0,4 pp","-1,7 pp","-0,9 pp","-16,1 pp","-6,4 pp","100,0%","-30,3 pp"],
 ["UAM","1,0 pp","-5,3 pp","2,3 pp","4,4 pp","-26,6 pp","-2,3 pp","12,1%","0,0 pp"],
 ["USJT","1,2 pp","-4,7 pp","1,8 pp","3,8 pp","-25,1 pp","-3,3 pp","11,6%","0,0 pp"],
 ["UNA","0,5 pp","-3,3 pp","1,9 pp","2,5 pp","-19,7 pp","-2,8 pp","11,7%","0,0 pp"],
 ["UNIBH","0,1 pp","-1,8 pp","-0,5 pp","1,4 pp","-33,4 pp","-4,2 pp","20,7%","0,0 pp"],
 ["IBMR","-6,6 pp","-2,0 pp","-0,7 pp","0,8 pp","-26,2 pp","-4,9 pp","109,0%","-49,5 pp"],
 ["FASEH","-1,2 pp","-9,9 pp","7,9 pp","8,8 pp","-11,6 pp","0,7 pp","7,3%","0,0 pp"],
 ["MIL. CAMPOS","7,1 pp","7,8 pp","-2,4 pp","-7,8 pp","0,0 pp","-7,5 pp","17,7%","0,0 pp"],
 ["UNISUL","-8,8 pp","-10,5 pp","6,9 pp","10,2 pp","-32,5 pp","5,7 pp","-4,2%","0,0 pp"],
 ["UNICURITIBA","3,7 pp","-2,5 pp","-0,4 pp","2,0 pp","-16,3 pp","-7,1 pp","13,5%","0,0 pp"],
 ["UNISOCIESC","-5,1 pp","-9,8 pp","6,8 pp","9,1 pp","-25,4 pp","9,3 pp","11,7%","0,0 pp"],
 ["UNR","0,0 pp","-1,5 pp","-1,9 pp","1,0 pp","-27,3 pp","-11,7 pp","88,4%","-47,5 pp"],
 ["FAD","2,7 pp","5,1 pp","-5,0 pp","-5,3 pp","-23,2 pp","-13,8 pp","80,9%","-26,5 pp"]
]
cols_deltas = ["MARCA","D_CONV","D_REND100","D_MENOR40","D_REPROV","D_NAO_AI","D_MIX_INAD","AUM","D_DEV_BOLSA"]
df_deltas = pd.DataFrame(raw_deltas, columns=cols_deltas)
for c in cols_deltas[1:]:
    df_deltas[c] = df_deltas[c].apply(to_float)

# ========= UI =========
st.title("Calouro - Pagante • Dispersões, Heatmaps e Barras")
st.caption("Aba 1: DELTAS (pp). Aba 2: Percentuais (%) + Heatmaps + Barras (IC95%). Cores = Δ % AUM.")

tab_deltas, tab_perc = st.tabs(["DELTAS (5 dispersões)", "Percentuais: 5 dispersões + Heatmaps + Barras"])

# ---- Aba DELTAS ----
with tab_deltas:
    st.subheader("DELTAS — X = Δ % Conversão (pp)")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            scatter(df_deltas, "D_CONV", "D_MENOR40",
                    "Δ % Conversão vs AA (pp)", "Δ % Média < 40 (pp)",
                    "Δ Conversão (X) vs Δ % Média < 40 (Y)"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            scatter(df_deltas, "D_CONV", "D_REPROV",
                    "Δ % Conversão vs AA (pp)", "Δ % Reprovado (pp)",
                    "Δ Conversão (X) vs Δ % Reprovado (Y)"),
            use_container_width=True
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            scatter(df_deltas, "D_CONV", "D_REND100",
                    "Δ % Conversão vs AA (pp)", "Δ % Rendimento 100% (pp)",
                    "Δ Conversão (X) vs Δ % Rendimento 100% (Y)"),
            use_container_width=True
        )
    with c4:
        st.plotly_chart(
            scatter(df_deltas, "D_CONV", "D_MIX_INAD",
                    "Δ % Conversão vs AA (pp)", "Δ % Mix Inadimplência (pp)",
                    "Δ Conversão (X) vs Δ % Mix Inadimplência (Y)"),
            use_container_width=True
        )

    st.plotly_chart(
        scatter(df_deltas, "D_CONV", "D_NAO_AI",
                "Δ % Conversão vs AA (pp)", "Δ % Não Realizou AI (pp)",
                "Δ Conversão (X) vs Δ % Não Realizou AI (Y)"),
        use_container_width=True
    )

# ---- Aba Percentuais ----
with tab_perc:
    st.subheader("Percentuais — X = % Conversão (nível atual)")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            scatter(df_perc, "P_CONV", "P_MENOR40",
                    "% Conversão (nível atual)", "% com Média < 40",
                    "% Conversão (X) vs % Média < 40 (Y)"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            scatter(df_perc, "P_CONV", "P_REPROV",
                    "% Conversão (nível atual)", "% Reprovado",
                    "% Conversão (X) vs % Reprovado (Y)"),
            use_container_width=True
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            scatter(df_perc, "P_CONV", "P_REND100",
                    "% Conversão (nível atual)", "% Rendimento 100%",
                    "% Conversão (X) vs % Rendimento 100% (Y)"),
            use_container_width=True
        )
    with c4:
        st.plotly_chart(
            scatter(df_perc, "P_CONV", "P_MIX_INAD",
                    "% Conversão (nível atual)", "% Mix Inadimplência",
                    "% Conversão (X) vs % Mix Inadimplência (Y)"),
            use_container_width=True
        )

    st.plotly_chart(
        scatter(df_perc, "P_CONV", "P_NAO_AI",
                "% Conversão (nível atual)", "% Não Realizou AI",
                "% Conversão (X) vs % Não Realizou AI (Y)"),
        use_container_width=True
    )

    st.markdown("### Heatmaps (%)")
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(
            heatmap_corr(df_perc, NUM_COLS_PERC, method="pearson", title_suffix="(Percentuais)"),
            use_container_width=True
        )
    with h2:
        st.plotly_chart(
            heatmap_corr(df_perc, NUM_COLS_PERC, method="spearman", title_suffix="(Percentuais)"),
            use_container_width=True
        )

    st.plotly_chart(
        barras_correlacoes(
            df_perc,
            x_var="P_CONV",
            var_map={
                "P_REND100": "REND",
                "P_MENOR40": "Média <40",
                "P_REPROV" : "Reprovação",
                "P_NAO_AI" : "NAO_AI",
                "P_MIX_INAD":"Inadimplência",
                "AUM"      : "AUM"
            },
            title="Correlações com % Conversão (IC 95%) — Percentuais"
        ),
        use_container_width=True
    )

st.markdown("---")
st.caption("Quadrantes em 0 (linhas tracejadas). Escala de cores: Viridis_r (Δ % AUM). Bases padrão embutidas.")

# =======================
# ABA 3 — ANÁLISE DESCRITIVA
# =======================
import io
from scipy.stats import spearmanr

def describe_frame(df, cols):
    out = []
    for c in cols:
        s = df[c].astype(float)
        out.append({
            "variável": c,
            "n": int(s.notna().sum()),
            "faltantes_%": 100 * s.isna().mean(),
            "média": s.mean(),
            "desvio": s.std(ddof=1),
            "min": s.min(),
            "q1": s.quantile(0.25),
            "mediana": s.median(),
            "q3": s.quantile(0.75),
            "máx": s.max(),
        })
    return pd.DataFrame(out)

def fisher_ci(r, n, alpha=0.05):
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha/2.0)
    return np.tanh(z - zcrit*se), np.tanh(z + zcrit*se)

def corr_table(df, x_var, y_vars, method="pearson"):
    rows = []
    for y in y_vars:
        sub = df[[x_var, y]].dropna().astype(float)
        n = len(sub)
        if n < 3:
            rows.append({"y": y, "n": n, "r/rho": np.nan, "lo95": np.nan,
                         "hi95": np.nan, "p_value": np.nan, "método": method})
            continue
        if method == "pearson":
            r, p = pearsonr(sub[x_var], sub[y])
            lo, hi = fisher_ci(r, n)
            rows.append({"y": y, "n": n, "r/rho": r, "lo95": lo, "hi95": hi,
                         "p_value": p, "método": "pearson"})
        else:
            rho, p = spearmanr(sub[x_var], sub[y])
            rows.append({"y": y, "n": n, "r/rho": rho, "lo95": np.nan,
                         "hi95": np.nan, "p_value": p, "método": "spearman"})
    return pd.DataFrame(rows)

def quadrant_counts(df, x_col, y_col):
    """Conta pontos por quadrante (linhas em 0)."""
    d = df[[x_col, y_col]].dropna().astype(float)
    q1 = ((d[x_col] > 0) & (d[y_col] > 0)).sum()
    q2 = ((d[x_col] <= 0) & (d[y_col] > 0)).sum()
    q3 = ((d[x_col] <= 0) & (d[y_col] <= 0)).sum()
    q4 = ((d[x_col] > 0) & (d[y_col] <= 0)).sum()
    return pd.DataFrame({"quadrante": ["Q1 (+,+)","Q2 (-,+)","Q3 (-,-)","Q4 (+,-)"],
                         "contagem": [q1,q2,q3,q4]})

def to_csv_download(df, filename):
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    st.download_button("⬇️ Baixar CSV: " + filename, buf.getvalue(), file_name=filename, mime="text/csv")

tab_desc, = st.tabs(["Análise Descritiva"])

with tab_desc:
    st.subheader("Análise Descritiva — Calouro")
    st.caption("Resumo, correlações, quadrantes (para DELTAS) e rankings rápidos.")

    # ---- Escolhas de variáveis
    st.markdown("##### Configurações")
    colA, colB = st.columns(2)
    with colA:
        x_deltas = "D_CONV"
        ylist_deltas = ["D_MENOR40","D_REPROV","D_REND100","D_MIX_INAD","D_NAO_AI"]
        st.write("**DELTAS:** X = `D_CONV` | Y = ", ", ".join(ylist_deltas))
    with colB:
        x_perc = "P_CONV"
        ylist_perc = ["P_MENOR40","P_REPROV","P_REND100","P_MIX_INAD","P_NAO_AI"]
        st.write("**Percentuais:** X = `P_CONV` | Y = ", ", ".join(ylist_perc))

    st.markdown("---")
    st.markdown("### 1) Estatísticas descritivas")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**DELTAS (pp)**")
        desc_deltas = describe_frame(df_deltas, ["D_CONV","D_REND100","D_MENOR40","D_REPROV","D_NAO_AI","D_MIX_INAD","AUM","D_DEV_BOLSA"])
        st.dataframe(desc_deltas, use_container_width=True)
        to_csv_download(desc_deltas, "descricao_deltas.csv")
    with col2:
        st.markdown("**Percentuais (%)**")
        desc_perc = describe_frame(df_perc, ["P_CONV","P_REND100","P_MENOR40","P_REPROV","P_NAO_AI","P_MIX_INAD","AUM","D_DEV_BOLSA"])
        st.dataframe(desc_perc, use_container_width=True)
        to_csv_download(desc_perc, "descricao_percentuais.csv")

    st.markdown("---")
    st.markdown("### 2) Correlações")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**DELTAS — Pearson (IC95%) com X = D_CONV**")
        ct_dp = corr_table(df_deltas, x_var=x_deltas, y_vars=ylist_deltas, method="pearson")
        st.dataframe(ct_dp, use_container_width=True)
        to_csv_download(ct_dp, "correlacoes_deltas_pearson.csv")
    with c2:
        st.markdown("**DELTAS — Spearman com X = D_CONV**")
        ct_ds = corr_table(df_deltas, x_var=x_deltas, y_vars=ylist_deltas, method="spearman")
        st.dataframe(ct_ds, use_container_width=True)
        to_csv_download(ct_ds, "correlacoes_deltas_spearman.csv")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Percentuais — Pearson (IC95%) com X = P_CONV**")
        ct_pp = corr_table(df_perc, x_var=x_perc, y_vars=ylist_perc, method="pearson")
        st.dataframe(ct_pp, use_container_width=True)
        to_csv_download(ct_pp, "correlacoes_percentuais_pearson.csv")
    with c4:
        st.markdown("**Percentuais — Spearman com X = P_CONV**")
        ct_ps = corr_table(df_perc, x_var=x_perc, y_vars=ylist_perc, method="spearman")
        st.dataframe(ct_ps, use_container_width=True)
        to_csv_download(ct_ps, "correlacoes_percentuais_spearman.csv")

    st.markdown("---")
    st.markdown("### 3) Quadrantes (apenas DELTAS, linhas em 0)")

    qcols = st.columns(5)
    for i, y in enumerate(ylist_deltas):
        with qcols[i]:
            st.markdown(f"**{y}**")
            qc = quadrant_counts(df_deltas, x_col="D_CONV", y_col=y)
            st.dataframe(qc, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 4) Rankings rápidos")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Top/Bottom por % Conversão (Percentuais)**")
        rank_conv = df_perc[["MARCA","P_CONV"]].dropna().sort_values("P_CONV", ascending=False).reset_index(drop=True)
        st.dataframe(rank_conv, use_container_width=True)
        to_csv_download(rank_conv, "ranking_percentuais_conv.csv")
    with r2:
        st.markdown("**Top/Bottom por Δ % Conversão (DELTAS)**")
        rank_dconv = df_deltas[["MARCA","D_CONV"]].dropna().sort_values("D_CONV", ascending=False).reset_index(drop=True)
        st.dataframe(rank_dconv, use_container_width=True)
        to_csv_download(rank_dconv, "ranking_deltas_dconv.csv")


