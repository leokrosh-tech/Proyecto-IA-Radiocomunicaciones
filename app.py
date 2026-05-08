import os
import glob
from io import BytesIO
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
import google.generativeai as genai

# ==========================================
# CONFIGURACIÓN DE SEGURIDAD
# ==========================================
# Puedes poner tu key en:
# 1) .streamlit/secrets.toml -> GOOGLE_API_KEY="TU_KEY"
# 2) Variable de entorno GOOGLE_API_KEY
# 3) Directo aquí, reemplazando PONE_TU_CLAVE_AQUI
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "PONE_TU_CLAVE_AQUI")

# ==========================================
# RUTAS DEL PROYECTO
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# CONFIGURACIÓN VISUAL
# ==========================================
st.set_page_config(page_title="IA Radiocomunicaciones UPB", page_icon="📡", layout="wide")

st.markdown(
    """
<style>
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00d2ff;
    }
    .titulo-pro {
        font-size: 40px;
        font-weight: 800;
        background: -webkit-linear-gradient(#00b4d8, #ffb703);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton > button {
        padding: 0px 5px !important;
    }
    .ok-box {
        background: #0f3d26;
        color: #65ff9a;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 600;
        margin-bottom: 14px;
        border: 1px solid rgba(101, 255, 154, 0.25);
    }
    .mini-title {
        font-size: 16px;
        font-weight: 700;
        margin: 16px 0 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="titulo-pro">📡 Simulador Predictivo de Radiocomunicaciones</p>', unsafe_allow_html=True)
st.markdown(
    "Plataforma empresarial enfocada en radioenlaces que combina **Data Science**, "
    "**Física ITU-R**, **gemelo digital 5-min** y **Google Gemini AI**."
)

# ==========================================
# NOMBRES CLAROS PARA TABLAS / EXCEL / GRÁFICAS
# ==========================================
RENOMBRES_DATOS = {
    "TIMESTAMP": "Fecha y hora",
    "RECORD": "Registro",
    "VBat": "Nivel Batería Eléctrica del Nodo (V)",
    "TempDL": "Carga Térmica Procesadores Data Logger (°C)",
    "PBar": "Presión Barométrica (mbar)",
    "TempA": "Temperatura Ambiental (°C)",
    "RH": "Humedad Relativa (%)",
}

COLUMNAS_BASE = ["TIMESTAMP", "RECORD", "VBat", "TempDL", "PBar", "TempA", "RH"]

# ==========================================
# MOTOR FÍSICO DE RADIOCOMUNICACIONES (ITU-R)
# ==========================================
def calcular_atenuacion_db(probabilidad, frecuencia_ghz, distancia_km):
    coeficientes = {
        10: (0.0101, 1.276),
        15: (0.0367, 1.154),
        20: (0.0751, 1.099),
        40: (0.3100, 0.929),
        80: (0.8606, 0.7656),
    }
    if frecuencia_ghz not in coeficientes:
        frecuencia_ghz = 15
    k, alpha = coeficientes[frecuencia_ghz]

    atenuacion_array = []
    for prob in probabilidad:
        if prob > 50:
            rain_rate = (prob / 100.0) * 40
            gamma = k * (rain_rate ** alpha)
            atenuacion_total = gamma * distancia_km
        else:
            atenuacion_total = 0.0
        atenuacion_array.append(atenuacion_total)
    return atenuacion_array

# ==========================================
# UTILIDADES DE ARCHIVOS Y EXPORTACIÓN
# ==========================================
def rutas_datasets():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")) + glob.glob(os.path.join(DATA_DIR, "*.dat")))


def guardar_archivos_permanentes(archivos):
    for file in archivos:
        path = os.path.join(DATA_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
    st.cache_data.clear()


def obtener_inventario_local():
    archivos = rutas_datasets()
    inventario = {}

    for ruta in archivos:
        nombre_archivo = os.path.basename(ruta)
        try:
            with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(4):
                    f.readline()
                linea_datos = f.readline()
                if linea_datos:
                    fecha_str = linea_datos.split(",")[0].replace('"', "")
                    ano = fecha_str[:4]
                    if not ano.isdigit():
                        ano = "Desconocidos"
                else:
                    ano = "Vacíos"
        except Exception:
            ano = "Errores"

        inventario.setdefault(ano, []).append({"ruta": ruta, "nombre": nombre_archivo})
    return inventario


def crear_excel_bytes(df, nombre_hoja="Datos"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=nombre_hoja[:31])
    return output.getvalue()


def preparar_tabla_5min(df):
    """Devuelve una tabla lista para mostrar y exportar con nombres claros."""
    out = df.copy()

    if "RECORD" not in out.columns:
        out.insert(1, "RECORD", np.arange(1, len(out) + 1))

    columnas_disponibles = [c for c in COLUMNAS_BASE if c in out.columns]
    out = out[columnas_disponibles].copy()

    if "TIMESTAMP" in out.columns:
        out["TIMESTAMP"] = pd.to_datetime(out["TIMESTAMP"], errors="coerce")
        out["TIMESTAMP"] = out["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M:%S")

    for col in ["VBat", "TempDL", "PBar", "TempA", "RH"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out.rename(columns=RENOMBRES_DATOS)


@st.cache_data(show_spinner=False)
def cargar_y_preparar_datos():
    dfs = []
    archivos = rutas_datasets()

    for archivo in archivos:
        try:
            df = pd.read_csv(archivo, skiprows=[0, 2, 3], low_memory=False)
            dfs.append(df)
        except Exception as e:
            st.warning(f"No se pudo leer {os.path.basename(archivo)}: {e}")

    if not dfs:
        return pd.DataFrame()

    data_completa = pd.concat(dfs, ignore_index=True)

    if "TIMESTAMP" not in data_completa.columns:
        st.error("No existe la columna TIMESTAMP en los datasets.")
        return pd.DataFrame()

    data_completa["TIMESTAMP"] = pd.to_datetime(data_completa["TIMESTAMP"], errors="coerce")
    data_completa.dropna(subset=["TIMESTAMP"], inplace=True)
    data_completa.sort_values("TIMESTAMP", inplace=True)

    for col in ["RH", "TempA", "PBar", "VBat", "TempDL"]:
        if col in data_completa.columns:
            data_completa[col] = pd.to_numeric(data_completa[col], errors="coerce")

    data_completa["Ano"] = data_completa["TIMESTAMP"].dt.year.astype(str)
    data_completa["Mes_Num"] = data_completa["TIMESTAMP"].dt.month
    data_completa["Mes"] = data_completa["TIMESTAMP"].dt.strftime("%m")
    data_completa["Dia"] = data_completa["TIMESTAMP"].dt.day
    data_completa["Hora"] = data_completa["TIMESTAMP"].dt.hour

    condiciones = [
        data_completa["Mes_Num"].isin([12, 1, 2]),
        data_completa["Mes_Num"].isin([3, 4, 5]),
        data_completa["Mes_Num"].isin([6, 7, 8]),
        data_completa["Mes_Num"].isin([9, 10, 11]),
    ]
    estaciones = ["☀️ Verano (Época Lluvia)", "🍂 Otoño", "❄️ Invierno (Época Seca)", "🌸 Primavera"]
    data_completa["Estacion"] = np.select(condiciones, estaciones, default="Desconocido")

    data_completa["Periodo_Mensual"] = data_completa["Ano"] + " - Mes " + data_completa["Mes"]
    data_completa["Periodo_Estacional"] = data_completa["Ano"] + " - " + data_completa["Estacion"]
    data_completa["Periodo_Anual"] = data_completa["Ano"]
    data_completa["Es_Lluvia"] = np.where(data_completa.get("RH", 0) > 85, 1, 0)

    return data_completa

# ==========================================
# MODELOS IA
# ==========================================
@st.cache_resource(show_spinner=False)
def entrenar_modelos_completos(df):
    columnas = ["Mes_Num", "Dia", "Hora", "Es_Lluvia", "TempA", "RH"]
    df_clean = df.dropna(subset=columnas).copy()
    if df_clean.empty:
        raise ValueError("No hay datos suficientes para entrenar los modelos predictivos.")

    X = df_clean[["Mes_Num", "Dia", "Hora"]]

    modelo_lluvia = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=14)
    modelo_lluvia.fit(X, df_clean["Es_Lluvia"])

    modelo_temp = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=14)
    modelo_temp.fit(X, df_clean["TempA"])

    modelo_rh = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=14)
    modelo_rh.fit(X, df_clean["RH"])

    return modelo_lluvia, modelo_temp, modelo_rh


def probabilidad_clase_1(modelo, X):
    """Evita error si el RandomForest solo vio una clase durante entrenamiento."""
    proba = modelo.predict_proba(X)
    if 1 in modelo.classes_:
        idx = list(modelo.classes_).index(1)
        return proba[:, idx] * 100
    return np.zeros(len(X))


@st.cache_resource(show_spinner=False)
def emular_hardware_termodinamico(df):
    columnas = ["TIMESTAMP", "TempA", "PBar", "VBat", "TempDL", "RH"]
    d_hw = df.dropna(subset=columnas).copy()
    if d_hw.empty:
        raise ValueError("No hay datos suficientes para entrenar el gemelo digital 5-min.")

    minutos = d_hw["TIMESTAMP"].dt.hour * 60 + d_hw["TIMESTAMP"].dt.minute
    d_hw["Sen_Dia"] = np.sin(2 * np.pi * minutos / 1440.0)
    d_hw["Cos_Dia"] = np.cos(2 * np.pi * minutos / 1440.0)

    dia_ano = d_hw["TIMESTAMP"].dt.dayofyear
    d_hw["Sen_Ano"] = np.sin(2 * np.pi * dia_ano / 365.25)
    d_hw["Cos_Ano"] = np.cos(2 * np.pi * dia_ano / 365.25)
    d_hw["Ano_Val"] = d_hw["TIMESTAMP"].dt.year

    X_cielo = d_hw[["Sen_Dia", "Cos_Dia", "Sen_Ano", "Cos_Ano", "Ano_Val"]]
    sky_ai = ExtraTreesRegressor(n_estimators=250, max_depth=30, random_state=42, n_jobs=-1)
    sky_ai.fit(X_cielo, d_hw[["TempA", "PBar"]])

    X_placa = X_cielo.copy()
    X_placa["TempA_Link"] = d_hw["TempA"]
    X_placa["Ciclo_Solar"] = np.where((d_hw["TIMESTAMP"].dt.hour >= 6) & (d_hw["TIMESTAMP"].dt.hour <= 18), 1, 0)

    hw_ai = ExtraTreesRegressor(n_estimators=300, max_depth=35, random_state=42, n_jobs=-1)
    hw_ai.fit(X_placa, d_hw[["VBat", "TempDL", "RH"]])

    return sky_ai, hw_ai

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Servidor Central y Red")
st.sidebar.markdown("Gestor de Data Loggers e Infraestructura")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Parámetros del Enlace RF")
freq_seleccionada = st.sidebar.selectbox("Frecuencia de Banda (GHz):", [10, 15, 20, 40, 80], index=1)
distancia_link = st.sidebar.slider("Distancia del enlace (Km):", 1.0, 50.0, 10.0, 0.5)

st.sidebar.info(
    "**💡 Significado:**\n\n"
    "🔹 **GHz:** frecuencia de transmisión. Frecuencias altas pueden transportar más datos, pero son más sensibles a lluvia.\n\n"
    "🔹 **Km:** distancia física entre antenas. A mayor distancia, más atmósfera atraviesa la señal."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Base de Datos en /data")
st.sidebar.caption(f"Carpeta leída: `{DATA_DIR}`")

archivos_subidos = st.sidebar.file_uploader("📂 Importar registros (.dat/.csv)", accept_multiple_files=True)
if archivos_subidos:
    if st.sidebar.button("💾 Guardar Datos Permanentemente"):
        guardar_archivos_permanentes(archivos_subidos)
        st.sidebar.success("Base de datos actualizada en /data.")
        st.rerun()

inventario_archivos = obtener_inventario_local()
if inventario_archivos:
    total_archivos = sum(len(v) for v in inventario_archivos.values())
    st.sidebar.success(f"🟢 Storage OK: **{total_archivos} registros** activos.")
    for ano in sorted(inventario_archivos.keys(), reverse=True):
        archivos_ano = inventario_archivos[ano]
        with st.sidebar.expander(f"📁 Año {ano} ({len(archivos_ano)} archivos)"):
            for arch in archivos_ano:
                col_n, col_b = st.columns([5, 1])
                col_n.markdown(f"<span style='font-size:12px;'>{arch['nombre']}</span>", unsafe_allow_html=True)
                if col_b.button("❌", key=f"del_{arch['ruta']}", help="Eliminar permanentemente"):
                    os.remove(arch["ruta"])
                    st.cache_data.clear()
                    st.rerun()
else:
    st.sidebar.error("🔴 Storage vacío. Coloca tus .dat/.csv en la carpeta /data.")

with st.spinner("Leyendo /data y construyendo matrices..."):
    df_main = cargar_y_preparar_datos()

# ==========================================
# APP PRINCIPAL
# ==========================================
if df_main.empty:
    st.warning("⚠️ No hay datos listos. Coloca archivos .dat/.csv dentro de la carpeta `data` o súbelos en la barra lateral.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analítica Histórica",
        "🔮 Presupuesto de Enlace Predictivo",
        "🖥️ Gemelo 5-Min + Excel",
        "🧠 Auditoría Directiva (IA)",
    ])

    # ------------------ PESTAÑA 1: ANALÍTICA HISTÓRICA ------------------
    with tab1:
        st.subheader("Auditoría de Microclima Operativo")
        col_f1, col_f2 = st.columns([1, 2])

        tipo_vista = col_f1.radio("Resolución de Filtro:", ["Mensual", "Estacional (Cochabamba)", "Anual"], horizontal=True)

        if tipo_vista == "Mensual":
            periodo_elegido = col_f2.selectbox("Seleccione el Segmento:", sorted(df_main["Periodo_Mensual"].dropna().unique()))
            df_plot = df_main[df_main["Periodo_Mensual"] == periodo_elegido]
        elif tipo_vista == "Estacional (Cochabamba)":
            periodo_elegido = col_f2.selectbox("Seleccione la Estación del Año:", sorted(df_main["Periodo_Estacional"].dropna().unique()))
            df_plot = df_main[df_main["Periodo_Estacional"] == periodo_elegido]
        else:
            periodo_elegido = col_f2.selectbox("Seleccione el Segmento:", sorted(df_main["Periodo_Anual"].dropna().unique()))
            df_plot = df_main[df_main["Periodo_Anual"] == periodo_elegido]

        st.markdown("##### 📈 Monitoreo de Instrumentos")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Temperatura Pico", f"{df_plot['TempA'].max():.1f} °C")
        kpi2.metric("Humedad Promedio", f"{df_plot['RH'].mean():.1f} %")
        kpi3.metric("Riesgos Documentados", f"{int(df_plot['Es_Lluvia'].sum())} eventos")
        kpi4.metric("Nodos Procesados", f"{len(df_plot):,}")

        freq = "W" if tipo_vista == "Anual" else "D"
        df_diario = df_plot.resample(freq, on="TIMESTAMP").agg({"TempA": "mean", "RH": "mean", "Es_Lluvia": "max"}).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_diario["TIMESTAMP"], y=df_diario["TempA"], name="Clima Térmico (°C)", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df_diario["TIMESTAMP"], y=df_diario["RH"], name="Saturación Humedad (%)", line=dict(width=3)))
        lluvias = df_diario[df_diario["Es_Lluvia"] == 1]
        fig.add_trace(go.Scatter(x=lluvias["TIMESTAMP"], y=lluvias["RH"], mode="markers", name="Riesgo >85% RH", marker=dict(size=10, symbol="x")))
        fig.update_layout(title=f"Telemetría en Radiocomunicaciones - {periodo_elegido}", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 Mostrar Mapa Científico de Correlaciones"):
            cols_corr = [c for c in ["TempA", "RH", "PBar", "VBat", "TempDL"] if c in df_plot.columns]
            if len(cols_corr) > 1:
                corr_matrix = df_plot[cols_corr].corr(numeric_only=True)
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No hay suficientes columnas numéricas para calcular correlación.")

    # ------------------ PESTAÑA 2: PRESUPUESTO DE ENLACE ------------------
    with tab2:
        st.subheader("Simulador Predictivo de Interrupciones (Link Budget)")

        col_p1, col_p2 = st.columns(2)
        tipo_prediccion = col_p1.selectbox("Horizonte de Predicción:", ["Proyección a Meses", "Proyección a Años"])
        horizonte = col_p2.slider("Tiempo al futuro:", 1, 12 if tipo_prediccion == "Proyección a Meses" else 5, 3)
        dias_futuro = (30 if tipo_prediccion == "Proyección a Meses" else 365) * horizonte

        if st.button("🚀 Iniciar Simulador Estocástico"):
            try:
                modelo_lluvia, modelo_temp, modelo_rh = entrenar_modelos_completos(df_main)

                fecha_fin = df_main["TIMESTAMP"].max() + timedelta(days=dias_futuro)
                fechas_futuras = pd.date_range(start=df_main["TIMESTAMP"].max(), end=fecha_fin, freq="h")

                df_f = pd.DataFrame({"TIMESTAMP": fechas_futuras})
                df_f["Mes_Num"] = df_f["TIMESTAMP"].dt.month
                df_f["Dia"] = df_f["TIMESTAMP"].dt.day
                df_f["Hora"] = df_f["TIMESTAMP"].dt.hour

                Xf = df_f[["Mes_Num", "Dia", "Hora"]]
                df_f["Prob_Alta"] = probabilidad_clase_1(modelo_lluvia, Xf)
                df_f["Semaforo_Riesgo"] = modelo_lluvia.predict(Xf)
                df_f["Temp_Pred"] = modelo_temp.predict(Xf)
                df_f["RH_Pred"] = modelo_rh.predict(Xf)
                df_f["Atenuacion_dB"] = calcular_atenuacion_db(df_f["Prob_Alta"], freq_seleccionada, distancia_link)

                df_res = df_f.resample("D", on="TIMESTAMP").agg({
                    "Prob_Alta": "max",
                    "Semaforo_Riesgo": "max",
                    "Temp_Pred": "mean",
                    "RH_Pred": "max",
                    "Atenuacion_dB": "max",
                }).reset_index()

                df_res["RH_Str"] = df_res["RH_Pred"].map("{:.1f}".format)
                df_res["Estado_Str"] = np.where(df_res["Semaforo_Riesgo"] == 1, "🚨 CRÍTICO", "✅ Operativo")
                custom_hover_data = np.stack((df_res["RH_Str"], df_res["Estado_Str"]), axis=-1)

                t_lluvias = int(df_res["Semaforo_Riesgo"].sum())
                perdida_max_db = df_res["Atenuacion_dB"].max()
                uptime_estimado = 100.0 - ((t_lluvias / len(df_res)) * 100)

                texto_guardado = (
                    f"Cálculo matemático predice que en los próximos {horizonte} periodos tendremos "
                    f"{t_lluvias} días críticos de lluvia extrema. A {freq_seleccionada} GHz y "
                    f"{distancia_link} km de radioenlace, esto generará caídas máximas de hasta "
                    f"{perdida_max_db:.2f} dB, dejando el Uptime estimado en {uptime_estimado:.2f}%. "
                    f"La temperatura promedio proyectada será {df_res['Temp_Pred'].mean():.1f} °C."
                )
                st.session_state["reporte_clima"] = texto_guardado
                st.session_state["datos_rf"] = df_res

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(label="Días de Fading Severo", value=f"{t_lluvias} Días", delta="Alertas de Red", delta_color="inverse")
                mc2.metric(label="Caída de Señal Máxima", value=f"{perdida_max_db:.1f} dB", delta=f"Física a {freq_seleccionada} GHz", delta_color="inverse")
                mc3.metric(label="Availability (Uptime)", value=f"{uptime_estimado:.2f} %", delta="SLA del Enlace")
                mc4.metric(label="Ruido Térmico Global", value=f"{df_res['Temp_Pred'].mean():.1f} °C", delta="Normal")

                st.markdown("### 📡 Escáner Predictivo de Perturbaciones (Dual-Axis)")
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])

                fig2.add_trace(
                    go.Scatter(
                        x=df_res["TIMESTAMP"],
                        y=df_res["Prob_Alta"],
                        name="Probabilidad de Lluvia",
                        fill="tozeroy",
                        mode="lines",
                        line=dict(width=3),
                        customdata=custom_hover_data,
                        hovertemplate=(
                            "<b>Prob. de Lluvia:</b> %{y:.1f}%<br>"
                            "<b>Humedad Max (RH):</b> %{customdata[0]}%<br>"
                            "<b>Estado Físico:</b> %{customdata[1]}<extra></extra>"
                        ),
                    ),
                    secondary_y=False,
                )

                fig2.add_trace(
                    go.Scatter(
                        x=df_res["TIMESTAMP"],
                        y=df_res["Atenuacion_dB"],
                        name="Pérdida de Señal (dB)",
                        mode="lines",
                        line=dict(width=4, dash="dash"),
                        hovertemplate="<b>Pérdida Calculada:</b> %{y:.2f} dB<extra></extra>",
                    ),
                    secondary_y=False,
                )

                fig2.add_trace(
                    go.Scatter(
                        x=df_res["TIMESTAMP"],
                        y=df_res["Temp_Pred"],
                        name="Temperatura (°C)",
                        mode="lines",
                        line=dict(width=3),
                        hovertemplate="<b>Temp. Atmosférica:</b> %{y:.1f} °C<extra></extra>",
                    ),
                    secondary_y=True,
                )

                dias_crit = df_res[df_res["Semaforo_Riesgo"] == 1]
                fig2.add_trace(
                    go.Scatter(
                        x=dias_crit["TIMESTAMP"],
                        y=dias_crit["Prob_Alta"],
                        mode="markers",
                        name="Fading Crítico Registrado",
                        marker=dict(size=14, symbol="triangle-down", line=dict(width=2)),
                        hoverinfo="skip",
                    ),
                    secondary_y=False,
                )

                fig2.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="rgba(10, 10, 10, 0.9)", font_size=15, font_family="Arial"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                )
                techo_dinamico = max(105, df_res["Atenuacion_dB"].max() + 15)
                fig2.update_yaxes(title_text="<b>Pico de Riesgo Diario (%) / Pérdida (dB)</b>", range=[0, techo_dinamico], secondary_y=False)
                fig2.update_yaxes(title_text="<b>Sensación Térmica (°C)</b>", secondary_y=True)
                st.plotly_chart(fig2, use_container_width=True)

                st.info(
                    "💡 **Cómo interpretar:**\n\n"
                    "🔹 La ola azul es la probabilidad máxima diaria de lluvia extrema.\n\n"
                    "🔹 La línea punteada representa pérdida de señal en dB.\n\n"
                    "🔹 La línea de temperatura ayuda a relacionar ambiente y riesgo."
                )

                tabla_rf = df_res.rename(columns={
                    "TIMESTAMP": "Fecha",
                    "Prob_Alta": "Probabilidad Alta Humedad/Lluvia (%)",
                    "Semaforo_Riesgo": "Alerta",
                    "Temp_Pred": "Temperatura Proyectada (°C)",
                    "RH_Pred": "Humedad Proyectada Máxima (%)",
                    "Atenuacion_dB": "Atenuación Estimada (dB)",
                })
                tabla_rf = tabla_rf[[
                    "Fecha",
                    "Probabilidad Alta Humedad/Lluvia (%)",
                    "Alerta",
                    "Temperatura Proyectada (°C)",
                    "Humedad Proyectada Máxima (%)",
                    "Atenuación Estimada (dB)",
                ]]

                c_dl1, c_dl2 = st.columns(2)
                c_dl1.download_button(
                    label="📥 Descargar Base Predictiva (CSV)",
                    data=tabla_rf.to_csv(index=False).encode("utf-8"),
                    file_name="Prediccion_Radiocomunicaciones_RF.csv",
                    mime="text/csv",
                )
                c_dl2.download_button(
                    label="📥 Descargar Base Predictiva (Excel)",
                    data=crear_excel_bytes(tabla_rf, "Prediccion_RF"),
                    file_name="Prediccion_Radiocomunicaciones_RF.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Error en simulación RF: {e}")

    # ------------------ PESTAÑA 3: GEMELO 5-MIN ------------------
    with tab3:
        st.subheader("🖥️ Gemelo digital 5-Min con tabla exportable")
        st.markdown(
            "Esta pestaña genera una lectura cada **5 minutos**, la muestra en tabla tipo datalogger "
            "y permite descargarla en **Excel** o **CSV**."
        )

        cxx, cvx = st.columns([1, 1])
        dia_eval = cxx.date_input("Fecha para simular / validar:", value=df_main["TIMESTAMP"].max().date())

        if cvx.button("🚀 Generar tabla 5-Min + gráfica"):
            try:
                with st.spinner("Entrenando gemelo y generando lecturas cada 5 minutos..."):
                    ia_cielo, ia_tierra = emular_hardware_termodinamico(df_main)
                    dtg_ref = df_main[df_main["TIMESTAMP"].dt.date == dia_eval].copy()

                    r_sim = pd.DataFrame({
                        "TIMESTAMP": pd.date_range(f"{dia_eval} 00:00:00", f"{dia_eval} 23:55:00", freq="5min")
                    })
                    minutos = r_sim["TIMESTAMP"].dt.hour * 60 + r_sim["TIMESTAMP"].dt.minute
                    dia_ano = r_sim["TIMESTAMP"].dt.dayofyear

                    Xd = pd.DataFrame({
                        "Sen_Dia": np.sin(2 * np.pi * minutos / 1440.0),
                        "Cos_Dia": np.cos(2 * np.pi * minutos / 1440.0),
                        "Sen_Ano": np.sin(2 * np.pi * dia_ano / 365.25),
                        "Cos_Ano": np.cos(2 * np.pi * dia_ano / 365.25),
                        "Ano_Val": r_sim["TIMESTAMP"].dt.year,
                    })

                    sky = ia_cielo.predict(Xd)
                    r_sim["TempA"] = sky[:, 0]
                    r_sim["PBar"] = sky[:, 1]

                    Xt = Xd.copy()
                    Xt["TempA_Link"] = r_sim["TempA"]
                    Xt["Ciclo_Solar"] = np.where((r_sim["TIMESTAMP"].dt.hour >= 6) & (r_sim["TIMESTAMP"].dt.hour <= 18), 1, 0)

                    placa = ia_tierra.predict(Xt)
                    r_sim["VBat"] = placa[:, 0]
                    r_sim["TempDL"] = placa[:, 1]
                    r_sim["RH"] = placa[:, 2]

                    # Corrección con el primer dato real disponible del día.
                    if not dtg_ref.empty:
                        anclaje_real = dtg_ref.iloc[0]
                        for col in ["VBat", "PBar", "TempA", "RH", "TempDL"]:
                            if col in anclaje_real and pd.notna(anclaje_real[col]):
                                r_sim[col] += anclaje_real[col] - r_sim[col].iloc[0]

                    # Suavizado y ruido leve para simular sensor.
                    for col in ["VBat", "TempDL", "PBar", "TempA", "RH"]:
                        r_sim[col] = r_sim[col].rolling(window=4, min_periods=1, center=True).mean()

                    rng = np.random.default_rng()
                    r_sim["VBat"] += rng.normal(0, 0.005, len(r_sim))
                    r_sim["TempDL"] += rng.normal(0, 0.03, len(r_sim))
                    r_sim["TempA"] += rng.normal(0, 0.04, len(r_sim))
                    r_sim["PBar"] += rng.normal(0, 0.03, len(r_sim))
                    r_sim["RH"] = np.clip(r_sim["RH"] + rng.normal(0, 0.3, len(r_sim)), 0, 100)

                    tabla_5min = preparar_tabla_5min(r_sim)
                    st.session_state["tabla_5min"] = tabla_5min
                    st.session_state["r_sim_5min"] = r_sim
                    st.session_state["dtg_ref_5min"] = dtg_ref
                    st.session_state["dia_eval_5min"] = dia_eval
            except Exception as e:
                st.error(f"Error al generar tabla 5-min: {e}")

        if "tabla_5min" in st.session_state and "r_sim_5min" in st.session_state:
            tabla_5min = st.session_state["tabla_5min"]
            r_sim = st.session_state["r_sim_5min"]
            dtg_ref = st.session_state.get("dtg_ref_5min", pd.DataFrame())
            dia_guardado = st.session_state.get("dia_eval_5min", dia_eval)

            st.markdown(
                '<div class="ok-box">💻 Compilación Multi-Data de Emulador Terminada — Ciclo diurno con frecuencia de actualización 5-min.</div>',
                unsafe_allow_html=True,
            )

            ult = r_sim.iloc[-1]
            temp_max = r_sim["TempA"].max()
            temp_min = r_sim["TempA"].min()
            temp_prom = r_sim["TempA"].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Batería Nodo", f"{ult['VBat']:.2f} V")
            m2.metric("Carga Térmica Data Logger", f"{ult['TempDL']:.2f} °C")
            m3.metric("Temperatura Ambiental Actual", f"{ult['TempA']:.2f} °C")
            m4.metric("Humedad Relativa", f"{ult['RH']:.2f} %")

            st.markdown("##### 🌡️ Resumen térmico del día")
            tx1, tx2, tx3 = st.columns(3)
            tx1.metric("Temp. Máxima del Día", f"{temp_max:.2f} °C")
            tx2.metric("Temp. Mínima del Día", f"{temp_min:.2f} °C")
            tx3.metric("Temp. Promedio del Día", f"{temp_prom:.2f} °C")

            st.markdown('<div class="mini-title">📋 Datos generados cada 5 minutos</div>', unsafe_allow_html=True)
            st.dataframe(tabla_5min, use_container_width=True, height=335, hide_index=True)

            col_excel, col_csv = st.columns([1, 1])
            col_excel.download_button(
                "📥 Descargar Excel (.xlsx)",
                data=crear_excel_bytes(tabla_5min, "Datos_5min"),
                file_name=f"datos_5min_{dia_guardado}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            col_csv.download_button(
                "📥 Descargar CSV",
                data=tabla_5min.to_csv(index=False).encode("utf-8"),
                file_name=f"datos_5min_{dia_guardado}.csv",
                mime="text/csv",
            )

            st.markdown('<div class="mini-title">Ciclos a nivel Silicio Interno sobre las placas cada 5-Min</div>', unsafe_allow_html=True)
            fig_hw = go.Figure()
            fig_hw.add_trace(go.Scatter(
                x=r_sim["TIMESTAMP"],
                y=r_sim["VBat"],
                name=RENOMBRES_DATOS["VBat"],
                mode="lines",
                hovertemplate="%{y:.2f} V<extra></extra>",
            ))
            fig_hw.add_trace(go.Scatter(
                x=r_sim["TIMESTAMP"],
                y=r_sim["TempDL"],
                name=RENOMBRES_DATOS["TempDL"],
                mode="lines",
                hovertemplate="%{y:.2f} °C<extra></extra>",
            ))
            fig_hw.add_trace(go.Scatter(
                x=r_sim["TIMESTAMP"],
                y=r_sim["TempA"],
                name=RENOMBRES_DATOS["TempA"],
                mode="lines",
                hovertemplate="%{y:.2f} °C<extra></extra>",
            ))
            fig_hw.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                height=430,
                legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98),
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_hw, use_container_width=True)

            if not dtg_ref.empty:
                with st.expander("🔎 Ver datos reales del día usado para validar"):
                    dtg_ref_suave = dtg_ref.copy()
                    for col in ["VBat", "TempDL", "PBar", "TempA", "RH"]:
                        if col in dtg_ref_suave.columns:
                            dtg_ref_suave[col] = dtg_ref_suave[col].rolling(3, min_periods=1, center=True).mean()
                    st.dataframe(preparar_tabla_5min(dtg_ref_suave), use_container_width=True, hide_index=True)
            else:
                st.info("La fecha seleccionada no tiene datos reales. Se generó una simulación completa sin anclaje físico del día.")

    # ------------------ PESTAÑA 4: GEMINI Y ACTA OFICIAL ------------------
    with tab4:
        st.subheader("Centro de Emisión de Dictámenes (Inteligencia Cognitiva)")
        st.markdown("Módulo alimentado por **Google Gemini** para redactar actas científicas basadas en la simulación RF.")

        if st.button("🧠 Compilar y Razonar Acta para el Jefe de Carrera"):
            if GOOGLE_API_KEY == "PONE_TU_CLAVE_AQUI" or GOOGLE_API_KEY == "":
                st.error("🔑 Falta la API Key de Google. Ponla en `.streamlit/secrets.toml`, variable de entorno o en la línea de configuración.")
            elif "reporte_clima" not in st.session_state:
                st.error("❌ Primero ejecuta la Pestaña 2 para generar datos de simulación RF.")
            else:
                with st.spinner("Sintetizando datos de radiofrecuencia con Gemini..."):
                    try:
                        genai.configure(api_key=GOOGLE_API_KEY)

                        modelo_objetivo = "models/gemini-2.5-flash"
                        for modelo in genai.list_models():
                            if "generateContent" in modelo.supported_generation_methods:
                                if "2.5" in modelo.name and "flash" in modelo.name.lower():
                                    modelo_objetivo = modelo.name
                                    break
                                if "2.5" in modelo.name and "pro" in modelo.name.lower():
                                    modelo_objetivo = modelo.name
                                    break

                        model = genai.GenerativeModel(modelo_objetivo)
                        prompt = f"""Eres el Ingeniero Jefe Analista de Datos y Redes de Radiocomunicación de la Universidad UPB en Bolivia.
Redacta un reporte ultra-profesional para el Director de la Carrera de Radiocomunicaciones basado en los resultados de nuestra IA:
DATOS CALCULADOS: {st.session_state['reporte_clima']}

Tu reporte debe contener ESTRICTAMENTE estas 4 secciones usando Markdown y viñetas elegantes:
### 1. Resumen Ejecutivo de Operaciones de Radioenlace
Sintetiza la situación de caídas de señal y Uptime estimado.
### 2. Análisis Físico de Atenuación (Rain Fading) en Microondas
Explica por qué ocurre la dispersión y atenuación de ondas con los decibeles calculados.
### 3. 💡 Factor Climatológico en Zonas de Alta Elevación
Incluye un dato científico sobre radiocomunicaciones en altura y valles de Bolivia.
### 4. Directrices Preventivas
Da 2 recomendaciones rápidas de mantenimiento o redundancia de espectro.

IMPORTANTE: No agregues firma ni despedida. Termina directamente en el punto 4.
"""
                        respuesta = model.generate_content(prompt)
                        st.session_state["documento_final"] = respuesta.text
                        st.success("✅ Acta Científica Compilada Exitosamente por Gemini.")
                    except Exception as e:
                        st.error(f"Falla crítica de IA o conexión: {e}")

        if "documento_final" in st.session_state:
            st.markdown(st.session_state["documento_final"])
            st.markdown("---")
            st.download_button(
                label="📄 Descargar Acta Oficial (.txt)",
                data=st.session_state["documento_final"],
                file_name="Dictamen_Radiocomunicaciones_UPB.txt",
                mime="text/plain",
            )
