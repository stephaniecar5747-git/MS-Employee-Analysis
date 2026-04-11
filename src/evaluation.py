import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Asegurar recursos
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def run_evaluation():
    # 1. Cargar datos
    if not os.path.exists("data/processed_en.csv"):
        print("❌ Error: No se encuentra el archivo procesado.")
        return
    
    df = pd.read_csv("data/processed_en.csv")
    vader = SentimentIntensityAnalyzer()

    # 2. Funciones de Sentimiento
    def get_textblob_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    # 3. Cálculo de métricas
    df['pros_vader'] = df['pros'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
    df['cons_vader'] = df['cons'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
    df['cons_textblob'] = df['cons'].apply(get_textblob_sentiment)

    # Etiquetas simplificadas
    df['sentiment_final'] = df['cons_vader'].apply(lambda x: 'Positivo' if x >= 0.05 else ('Negativo' if x <= -0.05 else 'Neutral'))

    # --- REPORTE IMPRESO EN CONSOLA ---
    print("\n" + "="*40)
    print("📊 RESUMEN DE MÉTRICAS DE LA EMPRESA")
    print("="*40)
    print(f"Total de reseñas analizadas: {len(df)}")
    print(f"Sentimiento promedio (Pros): {df['pros_vader'].mean():.2f}")
    print(f"Sentimiento promedio (Cons): {df['cons_vader'].mean():.2f}")
    print("\nDistribución de Sentimiento en Cons:")
    print(df['sentiment_final'].value_counts(normalize=True) * 100)
    print("="*40 + "\n")

    # --- VISUALIZACIONES ---
    plt.figure(figsize=(12, 5))

    # Gráfico 1: Distribución de Sentimientos (Barras)
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='sentiment_final', palette='viridis')
    plt.title('Distribución de Sentimiento (Cons)')
    plt.ylabel('Cantidad de Reseñas')

    # Gráfico 2: Comparación Pros vs Cons (Boxplot)
    plt.subplot(1, 2, 2)
    sentiment_data = df[['pros_vader', 'cons_vader']].melt(var_name='Tipo', value_name='Score')
    sns.boxplot(data=sentiment_data, x='Tipo', y='Score', palette='Set2')
    plt.title('Rango de Sentimiento: Pros vs Cons')
    
    plt.tight_layout()
    plt.savefig('data/reporte_visual.png') # Guarda la imagen para el MLOps
    print("📈 Gráfico guardado en 'data/reporte_visual.png'")
    plt.show()

    # 4. Guardar resultados
    df.to_csv("data/final_sentiment_report.csv", index=False)
    print("✅ Evaluación completa. Resultados guardados.")

if __name__ == "__main__":
    run_evaluation()