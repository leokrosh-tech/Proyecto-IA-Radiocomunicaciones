import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import timedelta
import google.generativeai as genai

# ==========================================
# CONFIGURACIÓN DE SEGURIDAD (BACKEND)
# ==========================================
# ==========================================
# CONFIGURACIÓN DE SEGURIDAD (BACKEND / NUBE)
# ==========================================
# La app intentará leer la clave secreta de la nube. Si estás en tu PC local, usará la tuya.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "PONE_TU_CLAVE_AQUI" # Pon tu clave aquí para cuando pruebes en tu PC

# ==========================================
# 1. CONFIGURACIÓN VISUAL Y CSS AVANZADO
# ==========================================
st.set_page_config(page_title="IA Radiocomunicaciones UPB", page_icon="📡", layout="wide")

st.markdown("""
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
    /* Hacemos que los botones de eliminar sean más sutiles */
    .stButton > button {
        padding: 0px 5px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="titulo-pro">📡 Simulador Predictivo de Radiocomunicaciones</p>', unsafe_allow_html=True)
st.markdown("Plataforma empresarial enfocada en radioenlaces que combina **Data Science (ML)**, **Física ITU-R P.838** y **Google Gemini (AI)** para el cálculo y mitigación del desvanecimiento de señal por lluvia (Rain Fade).")

# ==========================================
# 2. MOTOR FÍSICO DE RADIOCOMUNICACIONES (ITU-R)
# ==========================================
def calcular_atenuacion_db(probabilidad, frecuencia_ghz, distancia_km):
    coeficientes = {
        10: (0.0101, 1.276),   # Banda X
        15: (0.0367, 1.154),   # Banda Ku baja
        20: (0.0751, 1.099),   # Banda Ka baja
        40: (0.3100, 0.929),   # Banda V
        80: (0.8606, 0.7656)   # Banda E
    }
    
    if frecuencia_ghz not in coeficientes:
        frecuencia_ghz = 15

    k, alpha = coeficientes[frecuencia_ghz]
    
    atenuacion_array =[]
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
# 3. GESTOR DE ALMACENAMIENTO PERMANENTE Y NUBE
# ==========================================
if not os.path.exists('data'):
    os.makedirs('data')

def guardar_archivos_permanentes(archivos):
    for file in archivos:
        path = os.path.join('data', file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
    st.cache_data.clear()

def obtener_inventario_local():
    archivos = glob.glob('data/*.csv') + glob.glob('data/*.dat')
    inventario = {}
    for ruta in archivos:
        nombre_archivo = os.path.basename(ruta)
        try:
            with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(4): f.readline() 
                linea_datos = f.readline()
                if linea_datos:
                    fecha_str = linea_datos.split(',')[0].replace('"', '')
                    ano = fecha_str[:4] 
                    if not ano.isdigit():
                        ano = "Desconocidos"
                else:
                    ano = "Vacíos"
        except Exception:
            ano = "Errores"
            
        if ano not in inventario:
            inventario[ano] = []
        inventario[ano].append({"ruta": ruta, "nombre": nombre_archivo})
    return inventario

@st.cache_data
def cargar_y_preparar_datos():
    dfs =[]
    archivos = glob.glob('data/*.csv') + glob.glob('data/*.dat')
    for archivo in archivos:
        df = pd.read_csv(archivo, skiprows=[0, 2, 3], low_memory=False)
        dfs.append(df)
            
    if not dfs: return pd.DataFrame()

    data_completa = pd.concat(dfs, ignore_index=True)
    data_completa['TIMESTAMP'] = pd.to_datetime(data_completa['TIMESTAMP'], errors='coerce')
    data_completa.dropna(subset=['TIMESTAMP'], inplace=True)
    data_completa.sort_values('TIMESTAMP', inplace=True)
    
    for col in['RH', 'TempA', 'PBar']:
        if col in data_completa.columns:
            data_completa[col] = pd.to_numeric(data_completa[col], errors='coerce')

    data_completa['Ano'] = data_completa['TIMESTAMP'].dt.year.astype(str)
    data_completa['Mes_Num'] = data_completa['TIMESTAMP'].dt.month
    data_completa['Mes'] = data_completa['TIMESTAMP'].dt.strftime('%m')
    data_completa['Dia'] = data_completa['TIMESTAMP'].dt.day
    data_completa['Hora'] = data_completa['TIMESTAMP'].dt.hour
    
    condiciones =[
        data_completa['Mes_Num'].isin([12, 1, 2]),
        data_completa['Mes_Num'].isin([3, 4, 5]),
        data_completa['Mes_Num'].isin([6, 7, 8]),
        data_completa['Mes_Num'].isin([9, 10, 11])
    ]
    estaciones =['☀️ Verano (Época Lluvia)', '🍂 Otoño', '❄️ Invierno (Época Seca)', '🌸 Primavera']
    data_completa['Estacion'] = np.select(condiciones, estaciones, default='Desconocido')
    
    data_completa['Periodo_Mensual'] = data_completa['Ano'] + " - Mes " + data_completa['Mes']
    data_completa['Periodo_Estacional'] = data_completa['Ano'] + " - " + data_completa['Estacion']
    data_completa['Periodo_Anual'] = data_completa['Ano']
    
    data_completa['Es_Lluvia'] = np.where((data_completa['RH'] > 85), 1, 0)
    return data_completa

# ==========================================
# 4. INTELIGENCIA ARTIFICIAL MATEMÁTICA
# ==========================================
@st.cache_resource
def entrenar_modelos_duales(df):
    df_clean = df.dropna(subset=['Mes_Num', 'Dia', 'Hora', 'Es_Lluvia', 'TempA']).copy()
    X = df_clean[['Mes_Num', 'Dia', 'Hora']]
    
    y_lluvia = df_clean['Es_Lluvia']
    modelo_lluvia = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
    modelo_lluvia.fit(X, y_lluvia)
    
    y_temp = df_clean['TempA']
    modelo_temp = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
    modelo_temp.fit(X, y_temp)
    
    return modelo_lluvia, modelo_temp

# ==========================================
# 5. FRONTEND E INTERFAZ PRINCIPAL
# ==========================================
st.sidebar.header("⚙️ Servidor Central y Red")
st.sidebar.markdown("Gestor de Data Loggers e Infraestructura")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Parámetros del Enlace RF")
freq_seleccionada = st.sidebar.selectbox("Frecuencia de Banda (GHz):",[10, 15, 20, 40, 80], index=1, help="Banda X a Banda E")
distancia_link = st.sidebar.slider("Distancia del enlace (Km):", 1.0, 50.0, 10.0, 0.5)

st.sidebar.info(
    "**💡 ¿Qué significan estos valores?**\n\n"
    "🔹 **GHz (Frecuencia):** Es el 'tamaño' de la onda de radio. Frecuencias muy altas (como 40 u 80 GHz) permiten enviar datos rapidísimo (5G), pero son tan frágiles que **chocan con las gotas de lluvia** y se atenúan.\n\n"
    "🔹 **Km (Distancia):** Separación física entre ambas antenas. A mayor distancia, la señal debe atravesar más cantidad de lluvia en el aire, aumentando drásticamente la pérdida de señal."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Base de Datos en Servidor")

archivos_subidos = st.sidebar.file_uploader("📂 Importar registros (.dat)", accept_multiple_files=True)
if archivos_subidos:
    if st.sidebar.button("💾 Guardar Datos Permanentemente"):
        guardar_archivos_permanentes(archivos_subidos)
        st.sidebar.success("Base de datos actualizada en el servidor.")
        st.rerun()

inventario_archivos = obtener_inventario_local()

if inventario_archivos:
    total_archivos = sum([len(v) for v in inventario_archivos.values()])
    st.sidebar.success(f"🟢 Storage OK: **{total_archivos} registros** activos.")
    
    for ano in sorted(inventario_archivos.keys(), reverse=True):
        archivos_ano = inventario_archivos[ano]
        with st.sidebar.expander(f"📁 Año {ano} ({len(archivos_ano)} archivos)"):
            for arch in archivos_ano:
                col_nombre, col_boton = st.columns([5, 1])
                col_nombre.markdown(f"<span style='font-size:12px;'>{arch['nombre']}</span>", unsafe_allow_html=True)
                
                if col_boton.button("❌", key=f"del_{arch['ruta']}", help="Eliminar archivo permanentemente"):
                    os.remove(arch['ruta'])
                    st.cache_data.clear() 
                    st.rerun() 
else:
    st.sidebar.error("🔴 Storage Vacío.")

with st.spinner("Construyendo matrices neuronales..."):
    df_main = cargar_y_preparar_datos()

if df_main.empty:
    st.warning("⚠️ Esperando conexión de datos. Carga archivos en la barra lateral.")
else:
    tab1, tab2, tab3 = st.tabs(["📊 Analítica Histórica", "🔮 Presupuesto de Enlace Predictivo", "🧠 Auditoría Directiva (IA)"])
    
    # ------------------ PESTAÑA 1: ANALISIS ESTACIONAL ------------------
    with tab1:
        st.subheader("Auditoría de Microclima Operativo")
        col_f1, col_f2 = st.columns([1, 2])
        
        tipo_vista = col_f1.radio("Resolución de Filtro:",["Mensual", "Estacional (Cochabamba)", "Anual"], horizontal=True)
        
        if tipo_vista == "Mensual":
            periodo_elegido = col_f2.selectbox("Seleccione el Segmento:", sorted(df_main['Periodo_Mensual'].unique()))
            df_plot = df_main[df_main['Periodo_Mensual'] == periodo_elegido]
        elif tipo_vista == "Estacional (Cochabamba)":
            periodo_elegido = col_f2.selectbox("Seleccione la Estación del Año:", sorted(df_main['Periodo_Estacional'].unique()))
            df_plot = df_main[df_main['Periodo_Estacional'] == periodo_elegido]
        else:
            periodo_elegido = col_f2.selectbox("Seleccione el Segmento:", sorted(df_main['Periodo_Anual'].unique()))
            df_plot = df_main[df_main['Periodo_Anual'] == periodo_elegido]
        
        st.markdown("##### 📈 Monitoreo de Instrumentos")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Temperatura Pico", f"{df_plot['TempA'].max():.1f} °C")
        kpi2.metric("Humedad Constante Promedio", f"{df_plot['RH'].mean():.1f} %")
        kpi3.metric("Riesgos Documentados", f"{df_plot['Es_Lluvia'].sum()} eventos")
        kpi4.metric("Nodos Procesados", f"{len(df_plot):,}")

        freq = 'W' if tipo_vista == "Anual" else 'd'
        df_diario = df_plot.resample(freq, on='TIMESTAMP').agg({'TempA':'mean', 'RH':'mean', 'Es_Lluvia':'max'}).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_diario['TIMESTAMP'], y=df_diario['TempA'], name='Clima Térmico (°C)', line=dict(color='#ff9900', width=2)))
        fig.add_trace(go.Scatter(x=df_diario['TIMESTAMP'], y=df_diario['RH'], name='Saturación Humedad (%)', line=dict(color='#00d2ff', width=3)))
        lluvias = df_diario[df_diario['Es_Lluvia'] == 1]
        fig.add_trace(go.Scatter(x=lluvias['TIMESTAMP'], y=lluvias['RH'], mode='markers', name='Riesgo de Difracción (>85% RH)', marker=dict(color='red', size=10, symbol='x')))
        
        fig.update_layout(title=f"Telemetría en Radiocomunicaciones - {periodo_elegido}", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 Mostrar Mapa Científico de Correlaciones (Ingeniería de Datos)"):
            cols_corr =[c for c in['TempA', 'RH', 'PBar'] if c in df_plot.columns]
            if len(cols_corr) > 1:
                corr_matrix = df_plot[cols_corr].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------ PESTAÑA 2: PREDICCION FÍSICA Y GRÁFICO ------------------
    with tab2:
        st.subheader("Simulador Predictivo de Interrupciones (Link Budget)")
        
        col_p1, col_p2 = st.columns(2)
        tipo_prediccion = col_p1.selectbox("Horizonte de Predicción:",["Proyección a Meses", "Proyección a Años"])
        horizonte = col_p2.slider("Tiempo al futuro:", 1, 12 if tipo_prediccion == "Proyección a Meses" else 5, 3)
        dias_futuro = (30 if tipo_prediccion == "Proyección a Meses" else 365) * horizonte
        
        if st.button("🚀 Iniciar Simulador Estocástico"):
            modelo_lluvia, modelo_temp = entrenar_modelos_duales(df_main)
            
            fecha_fin = df_main['TIMESTAMP'].max() + timedelta(days=dias_futuro)
            fechas_futuras = pd.date_range(start=df_main['TIMESTAMP'].max(), end=fecha_fin, freq='h')
            
            df_f = pd.DataFrame({'TIMESTAMP': fechas_futuras})
            df_f['Mes_Num'] = df_f['TIMESTAMP'].dt.month
            df_f['Dia'] = df_f['TIMESTAMP'].dt.day
            df_f['Hora'] = df_f['TIMESTAMP'].dt.hour
            
            df_f['Prob_Alta'] = modelo_lluvia.predict_proba(df_f[['Mes_Num', 'Dia', 'Hora']])[:, 1] * 100
            df_f['Semaforo_Riesgo'] = modelo_lluvia.predict(df_f[['Mes_Num', 'Dia', 'Hora']])
            df_f['Temp_Pred'] = modelo_temp.predict(df_f[['Mes_Num', 'Dia', 'Hora']])
            
            df_f['Atenuacion_dB'] = calcular_atenuacion_db(df_f['Prob_Alta'], freq_seleccionada, distancia_link)
            
            df_res = df_f.resample('d', on='TIMESTAMP').agg({
                'Prob_Alta': 'max', 
                'Semaforo_Riesgo': 'max',
                'Temp_Pred': 'mean',
                'Atenuacion_dB': 'max' 
            }).reset_index()
            
            t_lluvias = df_res['Semaforo_Riesgo'].sum()
            f_lluvias = df_res[df_res['Semaforo_Riesgo']==1]['TIMESTAMP'].dt.strftime('%d/%m/%Y').tolist()
            prob_baja = df_res[df_res['Semaforo_Riesgo']==0]['Prob_Alta'].mean()
            perdida_max_db = df_res['Atenuacion_dB'].max()
            
            uptime_estimado = 100.0 - ((t_lluvias / len(df_res)) * 100)
            
            texto_guardado = f"Cálculo matemático predice que en los próximos {horizonte} periodos en Bolivia tendremos {t_lluvias} días críticos de lluvia extrema. A {freq_seleccionada} GHz y {distancia_link} km de radioenlace, esto generará caídas de señal máximas de hasta {perdida_max_db:.2f} dB, dejando el Uptime estimado en {uptime_estimado:.2f}%. Temperaturas promediarán {df_res['Temp_Pred'].mean():.1f}°C."
            st.session_state['reporte_clima'] = texto_guardado
            st.session_state['datos_csv'] = df_res 
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(label="Días de Fading Severo", value=f"{t_lluvias} Días", delta="Alertas de Red", delta_color="inverse")
            mc2.metric(label="Caída de Señal Máxima", value=f"{perdida_max_db:.1f} dB", delta=f"Física a {freq_seleccionada} GHz", delta_color="inverse")
            mc3.metric(label="Availability (Uptime)", value=f"{uptime_estimado:.2f} %", delta="SLA del Enlace")
            mc4.metric(label="Ruido Térmico Global", value=f"{df_res['Temp_Pred'].mean():.1f} °C", delta="Normal")
            
            st.markdown("### 📡 Escáner Predictivo de Perturbaciones (Dual-Axis)")
            
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig2.add_trace(go.Scatter(x=df_res['TIMESTAMP'], y=df_res['Prob_Alta'], name="Prob. Lluvia (%)", fill='tozeroy', mode='lines', line=dict(color='#00b4d8', width=3)), secondary_y=False)
            fig2.add_trace(go.Scatter(x=df_res['TIMESTAMP'], y=df_res['Atenuacion_dB'], name="Pérdida de Señal (dB)", mode='lines', line=dict(color='#9d4edd', width=4, dash='dash')), secondary_y=False)
            fig2.add_trace(go.Scatter(x=df_res['TIMESTAMP'], y=df_res['Temp_Pred'], name="Temperatura Prevista (°C)", mode='lines', line=dict(color='#ffaa00', width=3)), secondary_y=True)
            
            dias_crit = df_res[df_res['Semaforo_Riesgo'] == 1]
            fig2.add_trace(go.Scatter(x=dias_crit['TIMESTAMP'], y=dias_crit['Prob_Alta'], mode='markers', name='Fading Crítico Confirmado', marker=dict(color='#ff0033', size=14, symbol='triangle-down', line=dict(width=2, color='white'))), secondary_y=False)
            
            fig2.update_layout(template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
            
            techo_dinamico = max(105, df_res['Atenuacion_dB'].max() + 15)
            fig2.update_yaxes(title_text="<b>Pico de Riesgo Diario (%) / Pérdida (dB)</b>", range=[0, techo_dinamico], color="#00b4d8", secondary_y=False)
            fig2.update_yaxes(title_text="<b>Sensación Térmica (°C)</b>", color="#ffaa00", secondary_y=True)
            
            st.plotly_chart(fig2, use_container_width=True)

            st.info("💡 **Cómo interpretar los resultados técnicos:**\n\n"
                    "🔹 **La Ola Azul:** Es la probabilidad máxima diaria de que exista alta precipitación.\n\n"
                    "🔹 **La Línea Morada Punteada:** Es la atenuación física calculada en **Decibeles (dB)** usando el modelo matemático UIT-R. Si cruza el techo de 100, la gráfica escalará automáticamente.\n\n"
                    "🔹 **La Línea Dorada:** Temperatura del ambiente calculada de forma independiente.\n\n"
                    "🔻 **Triángulos Invertidos:** Marcan los días exactos donde se sugiere activar redundancia en la red por falla crítica inminente.")

            st.markdown("#### 💾 Extracción de Resultados Numéricos")
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar Predicciones de Radiocomunicaciones (CSV)", data=csv, file_name='Prediccion_Radiocomunicaciones_RF.csv', mime='text/csv')

    # ------------------ PESTAÑA 3: GEMINI Y ACTA OFICIAL ------------------
    with tab3:
        st.subheader("Centro de Emisión de Dictámenes (Inteligencia Cognitiva)")
        st.markdown("Módulo alimentado por **Google Gemini 2.5**. Capaz de razonar sobre la predicción matemática y redactar actas científicas para Radioenlaces.")
        
        if st.button("🧠 Compilar y Razonar Acta para el Jefe de Carrera"):
            if GOOGLE_API_KEY == "PONE_TU_CLAVE_AQUI" or GOOGLE_API_KEY == "":
                st.error("🔑 Alerta de Backend: Falta la API Key de Google en la línea 17.")
            elif 'reporte_clima' not in st.session_state:
                st.error("❌ Secuencia Errónea: Computa el futuro en la Pestaña 2 antes de invocar a la IA.")
            else:
                with st.spinner("Sintetizando datos de radiofrecuencia con la Base de Conocimiento de Google Cloud..."):
                    try:
                        genai.configure(api_key=GOOGLE_API_KEY)
                        
                        modelo_objetivo = 'models/gemini-2.5-flash'
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods:
                                if '2.5' in m.name and 'flash' in m.name.lower():
                                    modelo_objetivo = m.name
                                    break
                                elif '2.5' in m.name and 'pro' in m.name.lower():
                                    modelo_objetivo = m.name
                                    break
                                    
                        model = genai.GenerativeModel(modelo_objetivo)
                        
                        prompt = f"""Eres el Ingeniero Jefe Analista de Datos y Redes de Radiocomunicación de la Universidad UPB en Bolivia.
Redacta un reporte ultra-profesional para el Director de la Carrera de Radiocomunicaciones basado en los resultados de nuestra IA:
DATOS CALCULADOS: {st.session_state['reporte_clima']}

Tu reporte debe contener ESTRICTAMENTE estas 4 secciones usando Markdown y viñetas elegantes:
### 1. Resumen Ejecutivo de Operaciones de Radioenlace
(Sintetiza la situación de caídas de señal y Uptime estimado).
### 2. Análisis Físico de Atenuación (Rain Fading) en Microondas
(Explica como experto por qué ocurre la dispersión y atenuación de ondas con los decibeles que calculamos).
### 3. 💡 Factor Climatológico en Zonas de Alta Elevación (Curiosidad Científica)
(Escribe UN DATO CURIOSO y altamente científico sobre cómo la propagación electromagnética en radiocomunicaciones se comporta de forma especial en zonas de alta altitud y valles como en Bolivia, comparado con el nivel del mar).
### 4. Directrices Preventivas 
(2 recomendaciones rápidas de mantenimiento o redundancia de espectro).

IMPORTANTE: NO agregues ninguna firma, ni despedida, ni la palabra "Atentamente" al final del reporte. Termina directamente en el punto 4.
"""
                        respuesta = model.generate_content(prompt)
                        st.session_state['documento_final'] = respuesta.text
                        st.success("✅ Acta Científica Compilada Exitosamente por Gemini 2.5.")
                    except Exception as e:
                        st.error(f"Falla crítica de red neuronal: {e}")

        if 'documento_final' in st.session_state:
            with st.container():
                st.markdown(st.session_state['documento_final'])
            
            st.markdown("---")
            st.download_button(label="📄 Imprimir/Descargar Acta Oficial (.txt)", data=st.session_state['documento_final'], file_name="Dictamen_Radiocomunicaciones_UPB.txt", mime="text/plain")