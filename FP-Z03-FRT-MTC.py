# FP-Z05-FERT-MTC.py
import io
import re
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dashboard Fertilizantes - FarmPrecision",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#1b60a7",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#F1C40F",
    "info": "#E74C3C",
    "bg": "#F0F4F8",
}

EXPECTED_TOKENS = [
    "lote", "parcela", "lot", "codigo", "id_lote",
    "ano", "año", "year",
    "siembra", "planting",
    "material", "producto", "product", "fertilizante",
    "area", "ha", "has", "area_ha",
    "kg_n", "kg_n/ha", "kg_p", "kg_k", "p2o5", "k2o",
    "g_n", "g_p", "g_k",
    "ton", "tonelad"
]

def sanitize_key(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace("(", "").replace(")", "")

# ════════════════════════════════════════════════════
# CSS PERSONALIZADO
# ════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }

    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .kpi-sub   { font-size: 0.72rem; color: #7A8899; margin-top: 2px; }

    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }

    [data-testid="stSidebar"] { background: #F0F4F8; }

    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS
# ════════════════════════════════════════════════════

def sanitize_col_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^0-9a-zA-Z_áéíóúñÁÉÍÓÚÑ]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_').lower()

def _coerce_bytes_to_str(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode('utf-8', errors='ignore')
        except Exception:
            return str(x)
    return x

def smart_numeric_convert_series(s: pd.Series) -> pd.Series:

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s_obj = s.astype("object").map(
        lambda x: _coerce_bytes_to_str(x) if not pd.isna(x) else np.nan
    )

    def conv_token(x):
        if x is None:
            return np.nan
        if isinstance(x, float) and np.isnan(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)

        t = str(x).strip()
        if t == "":
            return np.nan

        t = t.replace("\xa0", "").replace(" ", "")
        t = re.sub(r"(?i)(kg|ha|ppm|cmol\(\+\)/kg|cmol/kg|%|mg/kg)", "", t)
        t = re.sub(r"[^0-9,\.\-]", "", t)

        if t in {"", "-", ".", ",", "-.", "-,"}:
            return np.nan

        if "," in t and "." in t:
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
        elif "," in t:
            if t.count(",") == 1:
                t = t.replace(",", ".")
            else:
                t = t.replace(",", "")

        try:
            return float(t)
        except Exception:
            return np.nan

    values = [conv_token(x) for x in s_obj]
    return pd.Series(values, index=s.index, name=s.name, dtype="float64")


def detect_header_row(df_preview: pd.DataFrame, n_top: int = 20) -> Tuple[int, List[str]]:
    rows = min(n_top, len(df_preview))
    best_row = 0
    best_score = -1
    best_values = []
    for i in range(rows):
        row_vals = df_preview.iloc[i].astype(object).astype(str).fillna('').tolist()
        row_lower = [r.strip().lower() for r in row_vals]
        tok_matches = 0
        non_empty = 0
        for cell in row_lower:
            if cell != '':
                non_empty += 1
            for tok in EXPECTED_TOKENS:
                if tok in cell:
                    tok_matches += 1
                    break
        score = tok_matches * 200 + non_empty
        if non_empty <= 2 and any(len(c) > 30 for c in row_lower):
            score -= 100
        if score > best_score:
            best_score = score
            best_row = i
            best_values = row_vals
    if best_score < 2:
        best_row = 0
        best_values = df_preview.iloc[0].astype(object).astype(str).fillna('').tolist()
        for i in range(1, rows):
            non_empty = df_preview.iloc[i].dropna().shape[0]
            if non_empty >= 3:
                best_row = i
                best_values = df_preview.iloc[i].astype(object).astype(str).fillna('').tolist()
                break
    return best_row, [str(v).strip() for v in best_values]

@st.cache_data(show_spinner=False)
def read_master_file_autodetect(uploaded, sample_rows: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    name = getattr(uploaded, "name", "") if hasattr(uploaded, "name") else str(uploaded)
    is_excel = isinstance(name, str) and name.lower().endswith((".xls", ".xlsx"))
    if not is_excel:
        raise ValueError("Este loader acepta únicamente archivos Excel (.xls, .xlsx).")
    if hasattr(uploaded, "read"):
        content = uploaded.read()
        try:
            uploaded.seek(0)
        except Exception:
            pass
    else:
        with open(str(uploaded), "rb") as f:
            content = f.read()

    xls = pd.ExcelFile(io.BytesIO(content))
    sheet_choice = None
    for s in xls.sheet_names:
        if re.search(r'aplic|aplica|aplicaciones|fertiliz|apli', s.lower()):
            sheet_choice = s
            break
    if sheet_choice is None:
        sheet_choice = xls.sheet_names[0]

    df0 = pd.read_excel(io.BytesIO(content), sheet_name=sheet_choice, header=None, dtype=object)
    header_row, _ = detect_header_row(df0, n_top=min(sample_rows, len(df0)))
    if header_row is None or header_row < 0:
        header_row = 0

    df_full = pd.read_excel(io.BytesIO(content), sheet_name=sheet_choice, header=None, dtype=object)
    if header_row >= len(df_full):
        header_row = 0

    raw_header = df_full.iloc[header_row].astype(object).astype(str).fillna('').tolist()
    df_data = df_full.iloc[header_row + 1:].copy().reset_index(drop=True)

    cleaned_header = []
    seen = {}
    for h in raw_header:
        h = "" if h is None else str(h).strip()
        if h == "":
            idx = seen.get("", 0) + 1
            seen[""] = idx
            h = f"unnamed_col_{idx}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 1
        cleaned_header.append(h)

    df_data.columns = cleaned_header
    return df_data, raw_header

def clean_dataframe_before_display(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = ["" if c is None else str(c) for c in d.columns]

    cols_to_drop = []
    for c in d.columns:
        try:
            non_null = d[c].notna().sum()
            total = max(len(d), 1)
            if non_null == 0:
                cols_to_drop.append(c)
            elif str(c).strip().lower().startswith("unnamed") and non_null / total < 0.05:
                cols_to_drop.append(c)
        except Exception:
            pass
    if cols_to_drop:
        d = d.drop(columns=cols_to_drop, errors="ignore")

    for c in list(d.columns):
        s = d[c]

        if pd.api.types.is_numeric_dtype(s):
            d[c] = pd.to_numeric(s, errors="coerce")
            continue

        try:
            s_obj = s.astype("object").map(
                lambda x: _coerce_bytes_to_str(x) if not pd.isna(x) else np.nan
            )
        except Exception:
            continue

        sample = s_obj.dropna().astype(str)
        if sample.empty:
            continue

        digit_ratio = sample.str.contains(r"\d", regex=True).mean()

        if digit_ratio > 0.20:
            num = smart_numeric_convert_series(s_obj)
            non_null_orig = s_obj.notna().sum()
            non_null_num  = num.notna().sum()
            success_ratio = non_null_num / max(non_null_orig, 1)

            if success_ratio >= 0.60:
                d[c] = num
            else:
                # mantener como texto limpio
                d[c] = s_obj.map(lambda x: x.strip() if isinstance(x, str) else x)
        else:
            d[c] = s_obj.map(lambda x: x.strip() if isinstance(x, str) else x)

    return d


def detect_columns_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    def find_any(patterns):
        for p in patterns:
            for c in cols:
                if p in str(c).lower():
                    return c
        return None
    mapping = {
        "lote":     ["lote","parcela","lot","codigo","id_lote"],
        "ano":      ["año","ano","year"],
        "siembra":  ["siembra","planting"],
        "material": ["material","producto","product","fertilizante"],
        "area":     ["area","ha","has","area_ha"],
        "kg_n_ha":  ["kg_n/ha","kg_n_ha","kg_n","n_kg_ha"],
        "kg_p_ha":  ["kg_p/ha","kg_p_ha","kg_p","p2o5","p_kg_ha"],
        "kg_k_ha":  ["kg_k/ha","kg_k_ha","kg_k","k_kg_ha"],
        "ton_mismo":["ton_ha_mismo","ton_mismo","ton/ha_mismo","toneladas_mismo"],
        "ton_post": ["ton_ha_posterior","ton_posterior","ton/ha_posterior"],
    }
    detected = {}
    for k, pats in mapping.items():
        detected[k] = find_any(pats)
    return detected

# ════════════════════════════════════════════════════
# 2. FUNCIONES DE GRÁFICOS
# ════════════════════════════════════════════════════

def compute_row_totals(df: pd.DataFrame, detected: Dict[str, Optional[str]]) -> pd.DataFrame:
    d = df.copy()

    def gcol(key):
        c = detected.get(key)
        return c if (c and c in d.columns) else None

    area_col = gcol("area")
    if area_col is None:
        for cand in ["area","area_ha","has","ha","area (ha)"]:
            matches = [c for c in d.columns if cand == str(c).strip().lower()]
            if matches:
                area_col = matches[0]
                break
    if area_col:
        d[area_col] = pd.to_numeric(d[area_col], errors="coerce")

    map_nutrients = {
        "N":    gcol("kg_n_ha") or next((c for c in d.columns if "kg_n" in str(c).lower()), None),
        "P2O5": gcol("kg_p_ha") or next((c for c in d.columns if "kg_p" in str(c).lower() or "p2o5" in str(c).lower()), None),
        "K2O":  gcol("kg_k_ha") or next((c for c in d.columns if "kg_k" in str(c).lower()), None),
        "Ca":   next((c for c in d.columns if "kg_ca" in str(c).lower()), None),
        "Mg":   next((c for c in d.columns if "kg_mg" in str(c).lower()), None),
        "S":    next((c for c in d.columns if "kg_s"  in str(c).lower()), None),
        "B":    next((c for c in d.columns if "kg_b"  in str(c).lower()), None),
        "Zn":   next((c for c in d.columns if "kg_zn" in str(c).lower()), None),
    }

    for short, col in map_nutrients.items():
        tot_col = f"{short}_kg_total"
        if col and col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            d[tot_col] = d[col] * d[area_col] if area_col else d[col]
        else:
            d[tot_col] = np.nan

    ton_m = gcol("ton_mismo")
    ton_p = gcol("ton_post")
    d["_ton_mismo_kg"] = np.nan
    d["_ton_post_kg"]  = np.nan
    if ton_m and area_col:
        d[ton_m] = pd.to_numeric(d[ton_m], errors="coerce")
        d["_ton_mismo_kg"] = d[ton_m] * d[area_col] * 1000.0
    if ton_p and area_col:
        d[ton_p] = pd.to_numeric(d[ton_p], errors="coerce")
        d["_ton_post_kg"] = d[ton_p] * d[area_col] * 1000.0

    nutrient_total_cols = [f"{k}_kg_total" for k in ["N","P2O5","K2O","Ca","Mg","S","B","Zn"]]

    def compute_proxy_row(r):
        if pd.notna(r.get("_ton_mismo_kg")) and r.get("_ton_mismo_kg") > 0:
            return r.get("_ton_mismo_kg")
        if pd.notna(r.get("_ton_post_kg")) and r.get("_ton_post_kg") > 0:
            return r.get("_ton_post_kg")
        s = 0.0; anyv = False
        for c in nutrient_total_cols:
            v = r.get(c)
            if pd.notna(v):
                try:
                    s += float(v); anyv = True
                except Exception:
                    pass
        if anyv and s > 0:
            return s
        for cand in ["kilos_aplicados","kilos","kg_aplic","kg_total"]:
            if cand in d.columns:
                v = r.get(cand)
                try:
                    if pd.notna(v): return float(v)
                except Exception:
                    pass
        return 1.0

    d["applied_mass_proxy_kg"] = d.apply(compute_proxy_row, axis=1)

    detected_again = detect_columns_map(d)
    for alias in ["lote","ano","material"]:
        c = detected_again.get(alias)
        if c and c in d.columns:
            std = "Lote" if alias == "lote" else ("Año" if alias == "ano" else "Material")
            if std not in d.columns:
                d.rename(columns={c: std}, inplace=True)

    if "Año" in d.columns:
        d["Año"] = pd.to_numeric(d["Año"], errors="coerce")

    return d


def plot_treemap_material(df: pd.DataFrame, detected: Dict[str, Optional[str]]):
    if df is None or df.empty:
        return go.Figure()
    d = df.copy()

    matcol = detected.get("material") if detected.get("material") in d.columns else \
             next((c for c in d.columns if "material" in str(c).lower()), None)
    lotcol = detected.get("lote") if detected.get("lote") in d.columns else \
             next((c for c in d.columns if "lote" in str(c).lower() or "parcela" in str(c).lower()), None)
    group_col = matcol or lotcol or (d.columns[0] if len(d.columns) > 0 else None)
    if group_col is None:
        return go.Figure()

    d[group_col] = d[group_col].astype(str).fillna("")
    if "applied_mass_proxy_kg" not in d.columns:
        d["applied_mass_proxy_kg"] = 0.0
    d["applied_mass_proxy_kg"] = pd.to_numeric(d["applied_mass_proxy_kg"], errors="coerce").fillna(0.0)

    agg = (
        d.groupby(group_col, dropna=False)
         .agg(mass_kg=("applied_mass_proxy_kg","sum"), records=(group_col,"count"))
         .reset_index()
         .sort_values("mass_kg", ascending=False)
    )
    agg = agg[agg["mass_kg"].fillna(0) > 0]

    if agg.empty:
        fallback = d[group_col].value_counts().reset_index().head(30)
        fallback.columns = [group_col, "count"]
        fig = px.bar(fallback, x="count", y=group_col, orientation="h",
                     color_discrete_sequence=[COLORS["primary"]],
                     title="Frecuencia de materiales (fallback)")
        fig.update_layout(height=360, margin=dict(t=30,l=0,r=0,b=0))
        return fig

    agg[group_col] = agg[group_col].str.replace(r'[\r\n]+',' ', regex=True).str.strip()
    try:
        labels  = agg[group_col].astype(str).tolist()
        parents = ["Materiales"] * len(labels)
        values  = agg["mass_kg"].astype(float).tolist()
        fig = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values,
            textinfo="label+value",
            marker=dict(colors=values, colorscale="Spectral")
        ))
        fig.update_layout(title=f"Uso (proxy kg) por {group_col}",
                          margin=dict(t=40,l=0,r=0,b=0), height=420)
        return fig
    except Exception:
        fig = px.bar(agg.head(30), x="mass_kg", y=group_col, orientation="h",
                     color_discrete_sequence=[COLORS["primary"]],
                     title=f"Uso (proxy kg) por {group_col} (fallback)")
        fig.update_layout(height=360, margin=dict(t=30,l=0,r=0,b=0))
        return fig


def plot_box_nutrients_perha(df: pd.DataFrame):
    traces = []
    perha_map = {
        "N":    ["kg_n/ha","kg_n_ha","kg_n"],
        "P2O5": ["kg_p/ha","kg_p_ha","kg_p","kg_p2o5"],
        "K2O":  ["kg_k/ha","kg_k_ha","kg_k"],
    }
    cols_lower = {str(c).lower(): c for c in df.columns}
    for label, pats in perha_map.items():
        found = None
        for p in pats:
            for lc, orig in cols_lower.items():
                if p in lc:
                    found = orig; break
            if found: break
        if found:
            vals = pd.to_numeric(df[found], errors="coerce").dropna()
            if not vals.empty:
                traces.append(go.Box(y=vals, name=f"{label} (kg/ha)", marker_color=COLORS["primary"]))
    fig = go.Figure(data=traces)
    fig.update_layout(title="Boxplots de nutrientes (kg/ha)", height=340,
                      margin=dict(t=40,l=0,r=0,b=0))
    return fig


def plot_top_lotes_by_nutrient(df: pd.DataFrame, detected: Dict[str, Optional[str]],
                                nutrient_short="N", topn=10):
    col    = f"{nutrient_short}_kg_total"
    lotcol = (detected.get("lote")     if detected.get("lote")     in df.columns else None) or \
             (detected.get("material") if detected.get("material") in df.columns else None)
    if col not in df.columns or lotcol is None:
        return go.Figure()
    agg = (
        df.groupby(lotcol, dropna=False)[col]
          .sum().reset_index()
          .sort_values(col, ascending=False)
          .head(topn)
    )
    if agg.empty:
        return go.Figure()
    agg["label"] = agg[lotcol].astype(str)
    fig = px.bar(agg, x=col, y="label", orientation="h",
                 color_discrete_sequence=[COLORS["primary"]],
                 title=f"Top {topn} {lotcol} por {nutrient_short} (kg total)")
    fig.update_layout(height=360, margin=dict(t=40,l=0,r=0,b=0))
    return fig


def plot_time_series_by_year(df: pd.DataFrame):
    cols_lower = {str(c).lower(): c for c in df.columns}
    ano_col = None
    for k in ["año","ano","year"]:
        if k in cols_lower:
            ano_col = cols_lower[k]; break

    totals = {f"{n}_kg_total": "sum" for n in ["N","P2O5","K2O"] if f"{n}_kg_total" in df.columns}
    if not totals:
        return go.Figure()

    d = df.copy()
    if ano_col:
        d[ano_col] = pd.to_numeric(d[ano_col], errors="coerce")
        agg = d.groupby(ano_col).agg(totals).reset_index().sort_values(ano_col)
        if agg.empty:
            return go.Figure()
        fig = px.bar(agg, x=ano_col, y=list(totals.keys()),
                     title="Aplicación anual por nutriente (kg)")
        fig.update_layout(height=420, margin=dict(t=40,l=0,r=0,b=0))
        return fig
    return go.Figure()

# ════════════════════════════════════════════════════
# 3. SECCIONES DEL DASHBOARD
# ════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    kpis = {"total_records": len(df)}
    for nutrient in ["N","P2O5","K2O"]:
        c = f"{nutrient}_kg_total"
        kpis[f"total_{nutrient}"] = (
            float(pd.to_numeric(df[c], errors="coerce").sum(skipna=True))
            if c in df.columns else 0.0
        )
    kpis["total_applied_mass_proxy_kg"] = (
        float(pd.to_numeric(df["applied_mass_proxy_kg"], errors="coerce").sum(skipna=True))
        if "applied_mass_proxy_kg" in df.columns else 0.0
    )
    return kpis


def seccion_kpis(df: pd.DataFrame):
    kpis = compute_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">N total (kg)</div>'
                f'<div class="kpi-value">{kpis["total_N"]:,.0f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">P₂O₅ total (kg)</div>'
                f'<div class="kpi-value">{kpis["total_P2O5"]:,.0f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">K₂O total (kg)</div>'
                f'<div class="kpi-value">{kpis["total_K2O"]:,.0f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Masa proxy (kg)</div>'
                f'<div class="kpi-value">{kpis["total_applied_mass_proxy_kg"]:,.0f}</div></div>', unsafe_allow_html=True)

def tab_overview(df: pd.DataFrame, detected: Dict[str, Optional[str]]):
    st.markdown("#### Mix de materiales y uso (proxy)")
    st.plotly_chart(plot_treemap_material(df, detected), use_container_width=True, key="treemap_material")

    st.markdown("#### Top materiales por proxy de masa")
    mat_col = detected.get("material") if detected.get("material") in df.columns else None
    if mat_col:
        agg = (
            df.groupby(mat_col, dropna=False)
              .agg(mass_kg=("applied_mass_proxy_kg","sum"))
              .reset_index()
              .sort_values("mass_kg", ascending=False)
        )
        st.dataframe(
            agg.head(50).rename(columns={mat_col:"Material","mass_kg":"Proxy_mass_kg"}),
            use_container_width=True, key="df_top_materials"
        )
    else:
        st.info("No se detectó columna 'Material'.")

def tab_application(df: pd.DataFrame, detected: Dict[str, Optional[str]]):
    st.markdown("#### Distribuciones kg/ha (boxplots)")
    st.plotly_chart(plot_box_nutrients_perha(df), use_container_width=True, key="box_nutrients")

    st.markdown("#### Top lotes por nutriente (kg total)")
    col1, col2 = st.columns([3, 1])
    with col2:
        topn        = st.number_input("Top N", min_value=3, max_value=50, value=10, key="topn_app")
        nutr_choice = st.selectbox("Nutriente", options=["N","P2O5","K2O"], index=0, key="nutr_choice_app")
    with col1:
        st.plotly_chart(
            plot_top_lotes_by_nutrient(df, detected, nutrient_short=nutr_choice, topn=topn),
            use_container_width=True,
            key=sanitize_key(f"top_lotes_{nutr_choice}_{topn}")
        )

def tab_temporal(df: pd.DataFrame):
    st.markdown("#### Series temporales (anual)")
    st.plotly_chart(plot_time_series_by_year(df), use_container_width=True, key="time_series_by_year")

def tab_table_export(df: pd.DataFrame):
    st.markdown("#### Tabla de registros")
    safe = clean_dataframe_before_display(df).head(200)
    st.dataframe(safe, use_container_width=True, height=400, key="table_sample")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV con columnas calculadas",
        csv,
        file_name="aplicaciones_fertilizantes_processed.csv",
        mime="text/csv",
        key="download_csv"
    )

# ════════════════════════════════════════════════════
# 4. SIDEBAR
# ════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 Dashboard Fertilizantes - FarmPrecision")

        st.markdown("---")
        st.markdown("### 📂 Cargar datos")
        uploaded = st.file_uploader(
            "Sube tu archivo Excel (.xlsx)",
            type=["xlsx", "xls"],
            help="El archivo debe contener columnas como finca, lote, año, siembra, material, área (ha) y columnas de nutrientes o toneladas.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")
        st.markdown("""
        | Interno |	Variantes aceptadas |
        |-------------|---------------------|
        | finca | finca, farm, hacienda |
        | lote | lote, lot, parcela, parcela_cod |
        | ano | ano, año, year |
        | siembra | siembra, año_siembra, planting |
        | material | material, producto, variedad |
        | area_ha | area_ha, area, ha |
        | kg N/ha | kg_n/ha, kg_n, n_kg_ha |
        | P2O5 | kg_p/ha, kg_p, p2o5 |
        | K2O | kg_k/ha, kg_k, k2o |
        | ton/ha | ton/ha, ton_ha, toneladas_ha |
        """)

    return uploaded

# ════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════

def main():
    uploaded = render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>🌴 Dashboard Fertilizantes — FarmPrecision</h1>
        <p>Análisis de insumos y nutrientes aplicados</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel para comenzar</h3>
            <p>Formatos soportados: <b>.xlsx</b> / <b>.xls</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    try:
        with st.spinner("⏳ Leyendo archivo y detectando encabezado..."):
            df_raw, original_columns = read_master_file_autodetect(uploaded)
            try:
                st.session_state["_last_uploaded_excel_name"] = getattr(uploaded, "name", "uploaded.xlsx")
                uploaded.seek(0)
                st.session_state["_last_uploaded_excel_bytes"] = uploaded.read()
            except Exception:
                pass
    except Exception as e:
        st.error(f"Error leyendo archivo Excel: {e}")
        st.stop()

    if df_raw is None or df_raw.empty:
        st.error("El archivo está vacío o no pudo ser leído como tabla.")
        st.stop()

    df_clean = clean_dataframe_before_display(df_raw)
    detected  = detect_columns_map(df_clean)
    df        = compute_row_totals(df_clean, detected)

    seccion_kpis(df)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧾 Overview",
        "📊 Aplicación",
        "📈 Temporal",
        "📋 Tabla & Export",
    ])

    with tab1:
        tab_overview(df, detected)
    with tab2:
        tab_application(df, detected)
    with tab3:
        tab_temporal(df)
    with tab4:
        tab_table_export(df)

if __name__ == "__main__":
    main()