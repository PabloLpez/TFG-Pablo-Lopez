# Detector Forense de Vídeo Generado por IA

> Sistema híbrido de detección de vídeo sintético (deepfakes y vídeo generado por modelos de difusión como Sora, Veo o Runway) que combina análisis forense de señal, machine learning supervisado, biometría facial y un modelo multimodal de lenguaje y visión.

**Trabajo de Fin de Grado — Grado en Tecnología Digital y Multimedia (GTDM), Universitat Politècnica de València.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-web%20app-black)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-green)

<!-- Recomendado: añade aquí una captura o GIF de la app en acción -->
<!-- ![Demo](docs/demo.gif) -->

## Índice

- [Sobre el proyecto](#sobre-el-proyecto)
- [Cómo funciona](#cómo-funciona)
- [Resultados](#resultados)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Ejecución](#ejecución)
- [Entrenar tus propios modelos (opcional)](#entrenar-tus-propios-modelos-opcional)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Sobre los datos](#sobre-los-datos)
- [Licencia](#licencia)
- [Autor](#autor)

## Sobre el proyecto

Con la llegada de modelos de generación de vídeo como Sora 2, Veo 3, Runway o Kling, distinguir a simple vista un vídeo real de uno sintético se ha vuelto extremadamente difícil. La mayoría de detectores comerciales son cajas negras entrenadas sobre datasets genéricos que no reflejan cómo se degrada un vídeo al pasar por Instagram, X/Twitter o WhatsApp.

Este proyecto aborda el problema con un sistema **explicable y sensible a la plataforma de origen**, que combina cuatro capas de análisis independientes:

- **Análisis forense de bajo nivel**: ~30 métricas extraídas de cada vídeo (nitidez, espectro de frecuencias, flujo óptico, ruido de sensor, aberración cromática, estructura de compresión I/P/B, metadatos) fundamentadas en literatura de forense digital.
- **Clasificación supervisada por plataforma**: modelos especializados (Random Forest / Gradient Boosting / XGBoost) para Instagram, Twitter/X y grabación directa, seleccionados automáticamente por un enrutador según el origen detectado, con un modelo unificado como respaldo.
- **Biometría facial complementaria**: cuando el vídeo contiene una cara estable, un segundo clasificador evalúa patrones de parpadeo, simetría facial y temblor de landmarks (MediaPipe Face Landmarker + 9 features dedicadas).
- **Razonamiento semántico**: análisis de coherencia visual con **Gemini 2.5 Flash** sobre frames representativos del vídeo.

Cada veredicto va acompañado de su explicación mediante **SHAP** (qué evidencias concretas lo justifican), en lugar de una única cifra sin contexto.

Demostración disponible en el archivo demostracion.mp4 del repositorio.

## Cómo funciona

```
Vídeo (archivo o URL) 
   → Detección de origen (Instagram / Twitter / YouTube / WhatsApp / móvil / desconocido)
   → Extracción de frames (FFmpeg, 5 fps)
   → Extracción de ~30 features forenses + metadatos (ExifTool)
   → Enrutador → modelo especializado o modelo unificado
   → (si hay cara estable) capa biométrica facial complementaria
   → Explicación SHAP de la predicción
   → Análisis de contexto semántico con Gemini
   → Informe final en la interfaz web (descargable en JSON)
```

## Resultados

Evaluación sobre un **holdout independiente de 139 vídeos**, nunca vistos durante el entrenamiento, procesados con el mismo pipeline que usa la aplicación:

<img width="1358" height="509" alt="image" src="https://github.com/user-attachments/assets/abf09a17-c434-429f-9c55-8174637accda" />

El desarrollo incluyó un diagnóstico explícito de **sesgo de dominio** entre plataformas (validación Leave-One-Platform-Out) que motivó la arquitectura de modelos especializados + enrutador.

<img width="1373" height="639" alt="image" src="https://github.com/user-attachments/assets/1c2933fc-7091-44e9-9388-cef413bbd5b9" />

## Stack tecnológico

| Área | Tecnologías |
|---|---|
| Backend | Python 3.11, Flask, Server-Sent Events |
| Visión por computador | OpenCV, FFmpeg / FFprobe |
| Machine Learning | scikit-learn, XGBoost, SHAP |
| Biometría facial | MediaPipe Face Landmarker |
| IA multimodal | Google Gemini API (2.5 Flash) |
| Metadatos forenses | ExifTool |
| Descarga de vídeo | yt-dlp |
| Frontend | HTML5 / CSS3 / JavaScript vanilla |

## Estructura del repositorio

```
.
├── app2.py                     # Servidor Flask — aplicación web
├── main42.py                   # Pipeline forense (clase AnalizadorVideo)
├── analisis_facial.py          # Extracción de features biométricas faciales
├── enrutador_modelos.py        # Selección de modelo según plataforma detectada
├── extraer_datos4.py           # Extracción masiva de features → CSV
├── extraer_caras.py            # Extracción masiva de features faciales → CSV
├── entrenar_ia5.py             # Entrenamiento y validación del clasificador general
├── entrenar_ia_facial.py       # Entrenamiento del clasificador facial
├── entrenar_especializados.py  # Entrenamiento de modelos por plataforma
├── testear_holdout_real.py     # Evaluación end-to-end sobre holdout independiente
├── static/
│   └── index3.html             # Frontend
├── modelo_v2.pkl               # Modelo unificado (pre-entrenado)
├── modelo_facial2.pkl          # Modelo facial (pre-entrenado)
├── especializados/              # Modelos especializados por plataforma + índice
├── requirements.txt
└── README.md
```

## Requisitos previos

- **Python 3.10 o superior** (recomendado 3.11)
- **FFmpeg y FFprobe** instalados y accesibles desde la terminal
- **ExifTool** instalado y accesible desde la terminal
- *(Opcional)* una clave de API gratuita de Google AI Studio para la capa de análisis de contexto con Gemini — la aplicación funciona sin ella, simplemente omite esa sección del informe

**Instalación de FFmpeg y ExifTool:**

```bash
# Windows (con Chocolatey)
choco install ffmpeg exiftool

# macOS (con Homebrew)
brew install ffmpeg exiftool

# Linux (Debian / Ubuntu)
sudo apt install ffmpeg libimage-exiftool-perl
```

Comprueba que están accesibles:

```bash
ffmpeg -version
ffprobe -version
exiftool -ver (si no está accesible, crear variable de entorno llamada EXIFTOOL_PATH cuya ruta apunte dentro de la carpeta con el .exe)
```

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>

# 2. Crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar las dependencias de Python
pip install -r requirements.txt
```

`requirements.txt` incluye:

```
flask
opencv-python
numpy
scipy
pandas
scikit-learn
xgboost
shap
matplotlib
yt-dlp
google-generativeai
mediapipe
python-dotenv
```

> La primera vez que se ejecuta el análisis facial, MediaPipe descarga automáticamente el modelo `face_landmarker.task` (~3 MB), por lo que se necesita conexión a internet en el primer uso.

## Ejecución

```bash
python app2.py
```

Abre el navegador en **http://localhost:5000**

Desde la interfaz puedes:
- Subir un vídeo (arrastrar y soltar) o pegar una URL de red social.
- Elegir el segmento a analizar si el vídeo dura más de 30 segundos.
- Forzar manualmente el origen o dejar que el sistema lo detecte automáticamente.
- Ver el veredicto principal, las evidencias forenses, el desglose SHAP, el panel biométrico facial (si hay cara detectada) y el análisis de contexto de Gemini (si está configurado).
- Descargar el informe completo en JSON.

## Entrenar tus propios modelos (opcional)

El repositorio incluye los modelos ya entrenados (`modelo_v2.pkl`, `modelo_facial2.pkl`, y los modelos especializados en `especializados/`), por lo que **no es necesario entrenar nada para probar la aplicación**. Si quieres reproducir o modificar el entrenamiento, necesitas tu propio dataset organizado así:

```
datasets/
├── real/
│   ├── instagram/
│   ├── twitter/
│   ├── movil_directo/
│   └── ...
└── fake/
    ├── instagram/
    ├── twitter/
    └── ...
```

Y ejecutar, en orden:

```bash
# 1. Extraer features del dataset general
python extraer_datos4.py --dir_real datasets/real --dir_fake datasets/fake --salida features_v2.csv

# 2. Entrenar el modelo unificado
python entrenar_ia5.py --csv features_v2.csv --salida modelo_v2.pkl

# 3. Entrenar modelos especializados por plataforma
python entrenar_especializados.py --csv features_v2.csv --modelo_unificado modelo_v2.pkl

# 4. (Opcional) Pipeline facial
python extraer_caras.py --dir_real datasets_facial/real --dir_fake datasets_facial/fake --salida features_facial.csv
python entrenar_ia_facial.py --csv features_facial.csv --salida modelo_facial2.pkl

# 5. Evaluar sobre un holdout independiente con el pipeline real
python testear_holdout_real.py --dir_holdout holdout --umbral 0.4
```

## Limitaciones y trabajo futuro

- El sistema está entrenado sobre un dataset de tamaño moderado; el rendimiento puede variar sobre generadores de vídeo muy recientes no representados en el entrenamiento.
- La capa forense pierde parte de su capacidad discriminativa en plataformas con recompresión agresiva (p. ej. Instagram).
- La capa biométrica facial es menos efectiva frente a los generadores más recientes, que ya reproducen parpadeo y micro-movimientos de forma razonablemente realista.
- No está pensado como prueba legal o forense definitiva, sino como herramienta de apoyo a la verificación.

## Sobre los datos

El dataset de vídeos (reales y generados por IA, algunos con rostros de personas) **no se incluye en este repositorio** por motivos de privacidad y derechos de imagen. La aplicación funciona igualmente porque **los modelos ya entrenados sí están incluidos**; solo hace falta un dataset propio si quieres reentrenar el sistema desde cero.

## Licencia

Distribuido bajo licencia MIT. Puedes usar, copiar y modificar el código citando la autoría original.

## Autor

**Pablo López Martínez**
Grado en Tecnología Digital y Multimedia — Universitat Politècnica de València
[LinkedIn](https://www.linkedin.com/in/pablo-lopez-martinez/) · [Correo](mailto:plopezm2004@gmail.com)
