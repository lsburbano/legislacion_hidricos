from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
import os

app = Flask(__name__)

# === Cliente OpenRouter para IA generativa ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # ← REEMPLAZA con tu clave real
)

# === Cargar modelos SARIMA ===
modelo_caudal = joblib.load("modelo_caudal.pkl")
modelo_temp = joblib.load("modelo_sarima_temperatura.pkl")
modelo_precipitacion = joblib.load("modelo_sarima_pre.pkl")

# === Cargar datos históricos ===
df_caudal = pd.read_excel("caudal_H34_H36_logico_corregido.xlsx", parse_dates=["Fecha"], index_col="Fecha")
serie_log = np.log(df_caudal["valor_H34"] + df_caudal["valor_H36"] + df_caudal["valor_H13"]).clip(lower=1)

df_temp = pd.read_csv("temperatura_mensual_papallacta_2025.csv")
df_temp = df_temp.replace(-999.0, np.nan).dropna()
fechas_temp = pd.to_datetime(df_temp[['YEAR', 'Month']].assign(DAY=1))
serie_temp = pd.Series(df_temp['Temperature_C'].astype(float).values, index=fechas_temp)

df_prec = pd.read_csv("precipitacion_mensual.csv")
fechas_prec = pd.to_datetime(df_prec[['YEAR', 'Month']].assign(DAY=1))
serie_precipitacion = pd.Series(df_prec['Precipitacion'].astype(float).values, index=fechas_prec)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.json
    fecha_pred = data["fecha"]  # formato "YYYY-MM"

    try:
        fecha_fin = pd.to_datetime(fecha_pred + '-01')

        # === Predicción de caudal ===
        meses_a_predecir_caudal = (fecha_fin.year - serie_log.index[-1].year) * 12 + (fecha_fin.month - serie_log.index[-1].month)
        if meses_a_predecir_caudal <= 0 or meses_a_predecir_caudal > 12:
            return jsonify({"error": "Selecciona un mes futuro válido (hasta 12 meses adelante)"}), 400

        pred_diff = modelo_caudal.forecast(steps=meses_a_predecir_caudal)
        pred_log = serie_log.iloc[-1] + pred_diff.cumsum()
        pred_caudal = np.exp(pred_log.iloc[-1])
        # Convertir m³/s a litros/s
        pred_caudal_lps = pred_caudal * 1000

        # === Predicción de temperatura ===
        meses_a_predecir_temp = (fecha_fin.year - serie_temp.index[-1].year) * 12 + (fecha_fin.month - serie_temp.index[-1].month)
        if meses_a_predecir_temp <= 0 or meses_a_predecir_temp > 12:
            pred_temp = "No disponible"
        else:
            pred_temp = modelo_temp.get_forecast(steps=meses_a_predecir_temp).predicted_mean.iloc[-1]
            pred_temp = round(pred_temp, 2)

        # === Predicción de precipitación ===
        meses_a_predecir_prec = (fecha_fin.year - serie_precipitacion.index[-1].year) * 12 + (fecha_fin.month - serie_precipitacion.index[-1].month)
        if meses_a_predecir_prec <= 0 or meses_a_predecir_prec > 12:
            pred_prec = "No disponible"
        else:
            pred_prec = modelo_precipitacion.get_forecast(steps=meses_a_predecir_prec).predicted_mean.iloc[-1]
            pred_prec = round(pred_prec, 2)*30

        poblacion_total = 1120000
        
        consumo_por_persona_dia = 180  # litros/día
        demanda_lps = (poblacion_total * consumo_por_persona_dia) / 86400

        # === Crear prompt para IA generativa ===
        prompt_text = f"""
        Genera un análisis técnico del riesgo hídrico en la zona de Papallacta con base en las variables de caudal, precipitación y temperatura. Organízalo en formato tipo infografía HTML, sin introducir el contenido ni explicar su propósito. Usa lenguaje claro y directo, sin términos meta como 'este bloque' o 'este análisis'. Solo devuelve el contenido embebible.
        Haz el análisis considerando el **conjunto** de variables climáticas y de oferta:
        - 💧 Caudal estimado: {pred_caudal_lps} l/s  
        - ☔ Precipitación mensual: {pred_prec} mm  
        - 🌡️ Temperatura media: {pred_temp} °C  

        Además, ten en cuenta:
        - 📅 Fecha de evaluación: {fecha_pred}  
        - 👥 Población atendida: 1,120,000 habitantes  
        - 🚿 Consumo per cápita: 180 l/día  
        -Demanda: 201,465,600 l/dia

        Evalúa cómo estas variables **interactúan entre sí** y qué implicaciones tienen en la disponibilidad de agua y el riesgo de escasez. No analices cada variable por separado, sino como un sistema conjunto que afecta la oferta y la sostenibilidad hídrica.

        Organiza el análisis en las siguientes secciones, usando títulos visuales con emojis:

        📊 Oferta y Demanda  
        ⚖️ Balance hídrico integrado  
        🌦️ Clima y factores externos  
        🚨 Riesgos y consecuencias  
        💡 Recomendaciones técnicas  
        📝 Conclusión ejecutiva
        Semaforo nivel:(rojo,verde,amarillo) segun tu analisis, siempre escribe el color 

        Usa un lenguaje claro y accesible para el público general,evita escribir esto como texto html, evitando tecnicismos complejos pero realiza los calculos necesarios que esto sea vea limpio y visualmente atractivo,
        """


        # === Solicitud al modelo IA en OpenRouter ===
        try:
            completion = client.chat.completions.create(
                model="openai/chatgpt-4o-latest",
                messages=[{"role": "user", "content": prompt_text}],
                extra_headers={
                    "HTTP-Referer": "https://mi-aplicacion.com",
                    "X-Title": "Prediccion Hidrica",
                },
                extra_body={},
            )
            analisis_ia = completion.choices[0].message.content

        except Exception as e:
            analisis_ia = f"Error al obtener análisis de IA: {str(e)}"

        return jsonify({
            "fecha": fecha_pred,
            "caudal_estimado": round(pred_caudal, 2)*1000,
            "temperatura": pred_temp,
            "precipitacion": pred_prec,
            "analisis_ia": analisis_ia
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Usa el puerto de Railway o 5000 por defecto
    app.run(host='0.0.0.0', port=port)