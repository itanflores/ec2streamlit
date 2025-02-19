import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import boto3

# 🛠️ Configurar la página
st.set_page_config(page_title=" Tablero de Monitoreo en Streamlit para la Gestión de Infraestructura TI", page_icon="📊", layout="wide")

# Ruta de salud para App Runner
if st.query_params.get("health") == "1":
    st.write("OK")
    exit(0)  # O usar return si está en una función
    
# 📢 Título del tablero
st.title("📊  Tablero de Monitoreo en Streamlit para la Gestión de Infraestructura TI")

# 🔹 Variables de entorno para AWS S3
S3_BUCKET = os.environ.get("S3_BUCKET", "tfm-monitoring-data")
S3_FILE = os.environ.get("S3_FILE", "dataset_procesado.csv")
LOCAL_FILE = "dataset_procesado.csv"

# 💞 Cargar Dataset desde S3
if not os.path.exists(LOCAL_FILE):
    s3 = boto3.client('s3')
    try:
        s3.download_file(S3_BUCKET, S3_FILE, LOCAL_FILE)
    except Exception as e:
        st.error(f"❌ Error: No se pudo descargar el dataset desde S3. Detalle: {e}")
        st.stop()

# 📌 Leer dataset con manejo de errores# 📌 Leer dataset con manejo de errores y validación de columnas
try:
    df = pd.read_csv(LOCAL_FILE)
    df.columns = df.columns.str.strip()
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors="coerce")

    # ⚠️ Verificar que las columnas esenciales estén presentes
    required_columns = {"Fecha", "Estado del Sistema", "Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)", "Temperatura (°C)"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        st.error(f"❌ Error: Faltan las siguientes columnas en el dataset: {', '.join(missing_columns)}")
        st.stop()

except Exception as e:
    st.error(f"❌ Error al leer el dataset: {e}")
    st.stop()

# 📌 Configuración de Streamlit en App Runner
if __name__ == "__main__":
    st.write("🚀 La aplicación está corriendo en AWS App Runner en el puerto 8080")

# 📌 Filtros
estados_seleccionados = st.multiselect("Selecciona uno o más Estados:", df["Estado del Sistema"].unique(), default=df["Estado del Sistema"].unique())
df_filtrado = df[df["Estado del Sistema"].isin(estados_seleccionados)]

# ⚠️ Manejo de datos vacíos
if df_filtrado.empty:
    st.warning("⚠️ No hay datos disponibles para los filtros seleccionados.")
    st.stop()

# 💫 Generar Datos de Estado
total_counts = df_filtrado["Estado del Sistema"].value_counts().reset_index()
total_counts.columns = ["Estado", "Cantidad"]

# 💫 Generar Datos de Estado Agrupados (con frecuencia diaria)
df_grouped = df_filtrado.groupby([pd.Grouper(key='Fecha', freq='D'), "Estado del Sistema"]).size().reset_index(name="Cantidad")

# ⚠️ Verificar si la columna "Cantidad_Suavizada" existe antes de continuar
try:
    df_grouped["Cantidad_Suavizada"] = df_grouped.groupby("Estado del Sistema")["Cantidad"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
except Exception as e:
    st.error(f"❌ Error al calcular 'Cantidad_Suavizada': {e}")
    st.stop()

try:
    df_avg = df_filtrado.groupby("Estado del Sistema")[["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"]].mean().reset_index()
except KeyError as e:
    st.error(f"❌ Error: Faltan columnas en el dataset. Detalle: {e}")
    st.stop()

# 🔹 Sección 1: Estado Actual
st.header("📌 Estado Actual")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Crítico", total_counts.loc[total_counts["Estado"] == "Crítico", "Cantidad"].values[0] if "Crítico" in total_counts["Estado"].values else 0)
kpi2.metric("Advertencia", total_counts.loc[total_counts["Estado"] == "Advertencia", "Cantidad"].values[0] if "Advertencia" in total_counts["Estado"].values else 0)
kpi3.metric("Normal", total_counts.loc[total_counts["Estado"] == "Normal", "Cantidad"].values[0] if "Normal" in total_counts["Estado"].values else 0)
kpi4.metric("Inactivo", total_counts.loc[total_counts["Estado"] == "Inactivo", "Cantidad"].values[0] if "Inactivo" in total_counts["Estado"].values else 0)

# ⚠️ Verificar si hay datos antes de asignar columnas
if df_grouped.empty:
    st.warning("⚠️ No hay datos disponibles después de aplicar los filtros.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(total_counts, values="Cantidad", names="Estado", title="📊 Distribución de Estados"), use_container_width=True)
        st.write("Este gráfico muestra la proporción de cada estado del sistema en el dataset. Útil para identificar tendencias y anomalías.")
        st.plotly_chart(px.bar(df_avg, x="Estado del Sistema", y=["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"], 
                               barmode="group", title="📊 Uso de Recursos"), use_container_width=True)
        st.write("Este gráfico compara el uso promedio de CPU, memoria y carga de red según el estado del sistema.")
    with col2:
        st.plotly_chart(px.line(df_grouped, x="Fecha", y="Cantidad_Suavizada", color="Estado del Sistema", 
                                title="📈 Evolución en el Tiempo", markers=True), use_container_width=True)
        st.write("Este gráfico representa la evolución temporal de los estados del sistema, permitiendo visualizar patrones y tendencias a lo largo del tiempo.")

        # 📊 Gráfico de dispersión: Relación entre Uso de CPU y Temperatura
        st.plotly_chart(px.scatter(
            df_filtrado,
            x="Uso CPU (%)",
            y="Temperatura (°C)",
            color="Estado del Sistema",
            title="📊 Relación entre Uso de CPU y Temperatura",
            labels={"Uso CPU (%)": "Uso de CPU (%)", "Temperatura (°C)": "Temperatura (°C)"},
            hover_name="Estado del Sistema"
        ), use_container_width=True)
        st.write("Este gráfico muestra la relación entre el uso de CPU y la temperatura, permitiendo identificar patrones y anomalías.")
    
# 🔹 Sección 2: Sección de Pronósticos
st.header("📈 Sección de Pronósticos")

# 📌 Predicción de Temperatura Crítica con normalización de datos
st.subheader("🌡️ Predicción de Temperatura Crítica")

if "Uso CPU (%)" in df_filtrado.columns and "Temperatura (°C)" in df_filtrado.columns:
    df_temp = df_filtrado[["Fecha", "Uso CPU (%)", "Carga de Red (MB/s)", "Temperatura (°C)"]].dropna()

    # ⚠️ Verificar que haya datos suficientes
    if df_temp.shape[0] < 10:
        st.warning("⚠️ No hay suficientes datos para predecir la temperatura crítica.")
    else:
        X = df_temp[["Uso CPU (%)", "Carga de Red (MB/s)"]]
        y = df_temp["Temperatura (°C)"]

        # 🔹 Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🔹 Entrenar modelo
        model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        model_temp.fit(X_scaled, y)

        # 🔹 Generar predicciones con valores normalizados
        future_data = pd.DataFrame({
            "Uso CPU (%)": np.linspace(X["Uso CPU (%)"].min(), X["Uso CPU (%)"].max(), num=12),
            "Carga de Red (MB/s)": np.linspace(X["Carga de Red (MB/s)"].min(), X["Carga de Red (MB/s)"].max(), num=12)
        })

        # 🔹 Transformar datos futuros con `StandardScaler`
        future_data_scaled = scaler.transform(future_data)
        future_temp_pred = model_temp.predict(future_data_scaled)

        df_future_temp = pd.DataFrame({
            "Fecha": pd.date_range(start=df_temp["Fecha"].max(), periods=12, freq="M"),
            "Temperatura Predicha (°C)": future_temp_pred
        })

        # 🔹 Graficar predicciones
        st.plotly_chart(px.line(df_future_temp, x="Fecha", y="Temperatura Predicha (°C)", 
                                title="📈 Predicción de Temperatura Crítica", markers=True), use_container_width=True)
        st.write("Este gráfico predice la temperatura crítica en función del uso de CPU y la carga de red.")


# 🔹 Nueva Sección: Análisis de Datos
st.header("📊 Análisis de Datos")

# 📊 Matriz de Correlación entre Variables
st.subheader("📊 Matriz de Correlación entre Variables")

required_columns_corr = ["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)", "Temperatura (°C)"]

if df_filtrado.empty or not all(col in df_filtrado.columns for col in required_columns_corr):
    st.warning("⚠️ No hay suficientes datos para calcular la matriz de correlación.")
else:
    # Calcular la matriz de correlación
    corr_matrix = df_filtrado[required_columns_corr].corr()

    # Mostrar la matriz de correlación como un heatmap
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Variable", y="Variable", color="Correlación"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="Viridis",
        title="Matriz de Correlación entre Variables"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.write("""
    Este gráfico muestra la matriz de correlación entre las variables del sistema. 
    - Un valor cercano a **1** indica una correlación positiva fuerte.
    - Un valor cercano a **-1** indica una correlación negativa fuerte.
    - Un valor cercano a **0** indica que no hay correlación.
    """)

# 🔹 Sección 3: Análisis de Outliers y Eficiencia Térmica
st.header("📊 Análisis de Outliers y Eficiencia Térmica")

# Evitar división por cero al calcular eficiencia térmica
uso_promedio_cpu = df_filtrado["Uso CPU (%)"].mean()
temperatura_media = df_filtrado["Temperatura (°C)"].mean()
eficiencia_termica = uso_promedio_cpu / temperatura_media if temperatura_media > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Uso Promedio de CPU (%)", f"{uso_promedio_cpu:.2f}")
col2.metric("Temperatura Media (°C)", f"{temperatura_media:.2f}")
col3.metric("Eficiencia Térmica", f"{eficiencia_termica:.2f}")

# Crear el Boxplot para Uso de CPU y Temperatura
st.subheader("📊 Distribución de Outliers (Boxplot)")
fig = px.box(df_filtrado, y=["Uso CPU (%)", "Temperatura (°C)"], title="Distribución de Uso de CPU y Temperatura")
st.plotly_chart(fig, use_container_width=True)
st.write("Este gráfico muestra la distribución de los valores de Uso de CPU y Temperatura, permitiendo identificar outliers y tendencias centrales.")

# Explicación de las métricas
st.write("""
- **Uso Promedio de CPU (%)**: Promedio del uso de CPU en el dataset filtrado.
- **Temperatura Media (°C)**: Promedio de la temperatura en el dataset filtrado.
- **Eficiencia Térmica**: Relación entre el uso de CPU y la temperatura. Un valor más alto indica mayor eficiencia.
""")

