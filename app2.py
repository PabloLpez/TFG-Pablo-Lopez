"""
app.py — Servidor web del analizador forense
--------------------------------------------
Importa AnalizadorVideo de main4.py (sin duplicar análisis).
El score principal es la probabilidad ML. Las explicaciones son SHAP.

Instalar:
    pip install flask yt-dlp shap

Ejecutar:
    python app.py  →  http://localhost:5000
"""

import os, json, uuid, threading, shutil, traceback, glob, types, base64
from pathlib import Path
from queue import Queue, Empty
from flask import Flask, request, jsonify, Response, send_from_directory, send_file
import subprocess, cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LimpiadorFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, sigma_clip=3.0):
        self.sigma_clip = sigma_clip

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(X, axis=0)
        X_imp = np.where(np.isnan(X), self.medians_, X)
        self.means_ = X_imp.mean(axis=0)
        self.stds_  = X_imp.std(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        mask = np.isnan(X)
        if mask.any():
            for j in range(X.shape[1]):
                X[mask[:, j], j] = self.medians_[j]
        for j in range(X.shape[1]):
            if self.stds_[j] > 0:
                lo = self.means_[j] - self.sigma_clip * self.stds_[j]
                hi = self.means_[j] + self.sigma_clip * self.stds_[j]
                X[:, j] = np.clip(X[:, j], lo, hi)
        return X
    
from main42 import AnalizadorVideo, FACTORES_ORIGEN
from enrutador_modelos import seleccionar_modelo

# ── Modelo facial (carga única al arrancar) ────────────────────
RUTA_MODELO_FACIAL = 'modelo_facial2.pkl'
_modelo_facial     = None
_fnames_facial     = None

def _cargar_modelo_facial():
    global _modelo_facial, _fnames_facial
    if not os.path.exists(RUTA_MODELO_FACIAL):
        print(f"[!] modelo_facial.pkl no encontrado — análisis facial desactivado.")
        return
    try:
        import pickle
        with open(RUTA_MODELO_FACIAL, 'rb') as f:
            d = pickle.load(f)
        _modelo_facial = d['modelo']
        _fnames_facial = d.get('feature_names', [
            'face_detectada','face_parpadeo_freq','face_parpadeo_simetria',
            'face_boca_apertura_std','face_landmark_temblor','face_eye_aspect_std',
            'face_simetria_facial','face_borde_estabilidad','face_iluminacion_consist',
        ])
        print(f"[+] Modelo facial cargado ({len(_fnames_facial)} features)")
    except Exception as e:
        print(f"[!] Error cargando modelo facial: {e}")

_cargar_modelo_facial()

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = 'AQ.Ab8RN6KiS5uc7r44L2YmaYlk6_L9ToCXWmfQ4uDOTF4aAS5Rmw'   # obtener gratis en aistudio.google.com
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    HAY_GEMINI = True
except ImportError:
    HAY_GEMINI = False
    print("[!] google-generativeai no instalado. Ejecuta: pip install google-generativeai")

app = Flask(__name__, static_folder='static')

RUTA_MODELO = 'modelo_v2.pkl'
DIR_TRABAJO = 'trabajo_tmp'
os.makedirs(os.path.join(DIR_TRABAJO, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(DIR_TRABAJO, 'thumbs'),  exist_ok=True)

tareas:  dict[str, Queue] = {}
videos:  dict[str, str]   = {}   # task_id → ruta del vídeo

DESCRIPCIONES = {
    'ruido_corr_temporal':
        "Correlación del ruido entre frames consecutivos. Una cámara real deja una huella "
        "persistente en el sensor que se repite de frame a frame. Si esta correlación es "
        "baja, el vídeo no tiene esa huella física.",
    'ruido_planitud_espectral':
        "Planitud del espectro de ruido. El ruido de un sensor real tiene más energía en "
        "frecuencias bajas (patrón 1/f). El ruido sintético es completamente uniforme en "
        "todas las frecuencias.",
    'ruido_asimetria':
        "Asimetría de la distribución del ruido. El ruido de sensor real no es perfectamente "
        "simétrico. Una asimetría casi nula indica que el ruido fue generado matemáticamente, "
        "no capturado por hardware.",
    'ruido_curtosis':
        "Curtosis del ruido. Las cámaras reales producen ruido con colas más pesadas de lo "
        "normal. Una curtosis casi nula es característica de ruido sintético perfectamente gaussiano.",
    'ruido_corr_canales':
        "Correlación del ruido entre canales de color. El proceso de interpolación de color "
        "de las cámaras (demosaicing) genera una correlación natural entre canales R y G. "
        "La IA no pasa por ese proceso.",
    'ruido_std_temporal':
        "Variación del nivel de ruido a lo largo del vídeo. El ruido de una cámara real varía "
        "con la exposición y el contenido. Un nivel de ruido perfectamente constante entre "
        "frames indica síntesis artificial.",
    'ruido_shapiro_p':
        "Test de gaussianidad del ruido. El ruido real de sensor no es perfectamente gaussiano. "
        "Si pasa el test con nota, el ruido es demasiado perfecto para ser real.",
    'ruido_std':
        "Nivel medio de ruido. Relacionado con la distribución estadística del residuo de "
        "ruido extraído de la imagen.",
    'meta_num_alertas':
        "Inconsistencias en los metadatos del archivo. Una grabación real contiene fabricante, "
        "modelo de cámara y fecha de grabación. Su ausencia o incoherencia puede indicar "
        "generación artificial.",
    'fft_ratio_std':
        "Variación de la distribución de frecuencias entre frames. Los vídeos reales cambian "
        "de escena a escena de forma natural. Una variación inusualmente alta puede indicar "
        "inconsistencia generada por IA.",
    'fft_energia_media':
        "Energía media del espectro de frecuencias. Valores muy altos pueden indicar sharpening "
        "artificial aplicado por un generador de IA o por la recodificación agresiva de "
        "plataformas sociales.",
    'fft_ratio_alta_baja':
        "Proporción de energía en altas vs bajas frecuencias. Las imágenes naturales tienen "
        "más energía en bajas frecuencias. Un exceso de altas frecuencias indica procesado artificial.",
    'ipb_pct_b':
        "Porcentaje de B-frames en el vídeo. Los codificadores reales los usan para comprimir "
        "mejor. Su ausencia total es típica de vídeos generados por IA o reexportados frame a frame.",
    'ipb_pct_i':
        "Porcentaje de I-frames (imágenes completas). En vídeos normales son menos del 5%. "
        "Un porcentaje alto indica que el codificador no aprovechó la redundancia temporal, "
        "como ocurre en vídeos de IA.",
    'ipb_pct_p':
        "Porcentaje de P-frames (diferencias con el frame anterior). Deben ser el tipo "
        "dominante. Un porcentaje anormalmente bajo indica que la codificación no siguió "
        "el patrón natural.",
    'ipb_gop_std':
        "Variación en el tamaño de los grupos de frames (GOP). Un codificador real adapta "
        "el GOP al contenido. Un GOP perfectamente regular indica una codificación mecánica "
        "sin adaptación.",
    'ipb_gop_medio':
        "Tamaño medio del grupo de frames. Un GOP muy corto indica que el codificador no "
        "está aprovechando la redundancia temporal entre frames, lo que es inusual en "
        "grabaciones reales.",
    'ipb_ratio_tamano_ip':
        "Diferencia de tamaño entre I-frames y P-frames. Los I-frames deben ser "
        "significativamente más pesados. Una ratio baja indica que los frames no "
        "contienen la información esperada.",
    'ipb_num_alertas':
        "Número total de anomalías detectadas en la estructura de compresión del vídeo. "
        "Agrupa los indicadores de I/P/B-frames, GOP y ratio de tamaño.",
    'gradiente_curtosis':
        "Curtosis de la distribución de bordes de la imagen. Las imágenes reales tienen "
        "bordes muy pronunciados en pocos puntos y suavidad en el resto. La IA tiende a "
        "distribuir los bordes de forma más uniforme.",
    'gradiente_entropia':
        "Uniformidad de las orientaciones de los bordes. Las imágenes reales tienen "
        "direcciones de borde preferentes. Una distribución demasiado uniforme de "
        "orientaciones indica generación artificial.",
    'flujo_suavidad':
        "Suavidad del movimiento entre frames. Una cámara real tiene pequeñas imperfecciones "
        "e inercia. Un movimiento anormalmente suave y predecible es típico de vídeos generados.",
    'flujo_salto_maximo':
        "Salto brusco máximo en el movimiento entre frames. Saltos muy altos sin un corte "
        "de plano que los justifique indican discontinuidades temporales artificiales.",
    'flujo_medio':
        "Magnitud media del movimiento a lo largo del vídeo. Valores inusualmente altos "
        "pueden indicar que el generador no controló correctamente la coherencia temporal.",
    'flujo_std':          "Variación del movimiento a lo largo del vídeo.",
    'flujo_num_glitches': "Número de anomalías estadísticas en el movimiento entre frames.",
    'nitidez_std':
        "Variación de nitidez entre frames. Una cámara real tiene variaciones naturales por "
        "el enfoque automático y el movimiento. Una variación muy alta o muy baja puede "
        "indicar síntesis.",
    'nitidez_outlier_pct':
        "Porcentaje de frames con nitidez muy diferente al resto. Un porcentaje alto indica "
        "que el generador no mantuvo coherencia de foco a lo largo del vídeo.",
    'nitidez_media': "Nivel medio de nitidez de los frames del vídeo.",
    'brillo_std':
        "Variación del brillo entre frames. Cambios bruscos de exposición que no "
        "corresponden a cambios reales en la escena pueden indicar iluminación generada artificialmente.",
    'brillo_tendencia':
        "Tendencia lineal del brillo a lo largo del vídeo. Una tendencia muy pronunciada "
        "puede indicar un fade artificial aplicado por el generador.",
    'luz_std':
        "Variación de la dirección de iluminación entre frames. Cambios bruscos en la "
        "dirección de la luz sin justificación física indican inconsistencias de una escena generada.",
    'ca_desfase_std':
        "Uniformidad de la aberración cromática entre zonas de la imagen. Las lentes reales "
        "tienen más aberración en los bordes que en el centro. Una aberración perfectamente "
        "uniforme indica lente sintética.",
    'ca_desfase_medio':
        "Nivel medio de aberración cromática. El desplazamiento entre los canales de color "
        "que produce cualquier lente óptica real.",
    'dct_ratio':
        "Patrón de compresión en bloques. Un ratio alto indica que el vídeo ha sido "
        "comprimido y recomprimido múltiples veces, dejando artefactos en la cuadrícula.",
    'corr_rg':
        "Correlación entre los canales rojo y verde. Valores muy altos indican que los "
        "canales son casi idénticos, lo que puede sugerir generación sin variación cromática real.",
    'corr_rb': "Correlación entre los canales rojo y azul.",
    'corr_gb': "Correlación entre los canales verde y azul.",
}


# ════════════════════════════════════════════════════════════════
#  ANÁLISIS DE CONTEXTO — GEMINI 2.5 FLASH
# ════════════════════════════════════════════════════════════════

def analizar_contexto_gemini(dir_frames, total_frames):
    """
    Manda 4 frames representativos a Gemini 2.5 Flash y pide
    un análisis del contexto visual del vídeo.
    Devuelve None si Gemini no está disponible o falla.
    """
    if not HAY_GEMINI or not GEMINI_API_KEY or GEMINI_API_KEY == 'TU_CLAVE_AQUI':
        return None

    try:
        # Seleccionar 4 frames repartidos a lo largo del vídeo
        indices = sorted(set([
            1,
            max(1, total_frames // 3),
            max(1, total_frames * 2 // 3),
            total_frames,
        ]))

        partes = []
        for idx in indices:
            p = os.path.join(dir_frames, f'out{idx}.png')
            if not os.path.exists(p):
                continue
            with open(p, 'rb') as f:
                datos = base64.b64encode(f.read()).decode('utf-8')
            partes.append({'inline_data': {'mime_type': 'image/png', 'data': datos}})

        if not partes:
            return None

        partes.append({'text': """Analiza estas imágenes extraídas de un vídeo y responde SOLO en español.
Tu objetivo es evaluar si el contenido visual podría ser real o generado por inteligencia artificial.

Responde con este formato exacto, una línea por campo:
CONTEXTO: [describe brevemente qué muestra el vídeo en 1-2 frases]
COHERENCIA: [¿es físicamente posible lo que se ve? ¿hay inconsistencias en iluminación, proporciones o física?]
ARTEFACTOS: [¿hay bordes extraños, texturas artificiales, manos/caras deformadas, texto ilegible u otros signos típicos de IA?]
VEREDICTO_CONTEXTO: [escribe solo una de estas opciones: PROBABLE_IA / AMBIGUO / PROBABLE_REAL]
CONFIANZA: [escribe solo una de estas opciones: ALTA / MEDIA / BAJA]
RAZON: [1 frase corta explicando el veredicto]"""
        })

        modelo   = genai.GenerativeModel('gemini-2.5-flash')
        respuesta = modelo.generate_content(partes)
        return parsear_respuesta_gemini(respuesta.text)

    except Exception as e:
        print(f"    [!] Gemini error: {e}")
        return {'error': str(e)}


def parsear_respuesta_gemini(texto):
    resultado = {
        'texto_completo': texto,
        'contexto':  '', 'coherencia': '', 'artefactos': '',
        'veredicto': '', 'confianza':  '', 'razon':      '',
    }
    mapeo = {
        'CONTEXTO:':           'contexto',
        'COHERENCIA:':         'coherencia',
        'ARTEFACTOS:':         'artefactos',
        'VEREDICTO_CONTEXTO:': 'veredicto',
        'CONFIANZA:':          'confianza',
        'RAZON:':              'razon',
    }
    for linea in texto.split('\n'):
        linea = linea.strip()
        for clave, campo in mapeo.items():
            if linea.upper().startswith(clave):
                resultado[campo] = linea[len(clave):].strip()
    return resultado


# ════════════════════════════════════════════════════════════════
#  PIPELINE
# ════════════════════════════════════════════════════════════════

def ejecutar_analisis(ruta_video, dir_frames, q: Queue,
                      segundos: int = None, task_id: str = None,
                      origen_manual: str = 'auto',
                      segundo_inicio: int = 0):
    def emit(tipo, **kw): q.put({'type': tipo, **kw})
    def step(label, pct): emit('progress', label=label, pct=pct)

    try:
        a = AnalizadorVideo.__new__(AnalizadorVideo)
        a.ruta              = ruta_video
        a.informe           = {}
        a.frames            = []
        a.frame_color       = None
        a.total_frames      = 0
        a.segundos_analizar = segundos
        a._dir              = dir_frames
        os.makedirs(dir_frames, exist_ok=True)

        def _ruta_frame(self, i): return os.path.join(self._dir, f"out{i}.png")
        def _limpiar(self):
            for f in glob.glob(os.path.join(self._dir, 'out*.png')): os.remove(f)
        a._ruta_frame    = types.MethodType(_ruta_frame, a)
        a.limpiar_frames = types.MethodType(_limpiar,    a)

        step('Detectando origen…', 5)
        a.detectar_origen()

        # Si el usuario eligió manualmente un origen distinto de 'auto', lo sobreescribe
        if origen_manual and origen_manual != 'auto':
            a.informe['origen_detectado'] = origen_manual
            print(f"  [+] Origen manual: {origen_manual}")

        # Enrutar al modelo apropiado según origen final
        origen_final = a.informe.get('origen_detectado', 'desconocido')
        ruta_modelo_usar, tipo_modelo = seleccionar_modelo(origen_final, fallback=RUTA_MODELO)
        step(f'Cargando modelo {tipo_modelo} ({origen_final})…', 9)
        a.cargar_modelo(ruta_modelo_usar)

        step('Analizando metadatos…', 13)
        a.analizar_metadatos()
        a.analizar_metadatos_forenses()

        # Construir argumentos de tiempo para ffmpeg
        # Soporta: completo / primeros Ns / fragmento SS→SS+dur
        args_tiempo = []
        if segundo_inicio and segundo_inicio > 0:
            args_tiempo += ['-ss', str(segundo_inicio)]
        if segundos:
            args_tiempo += ['-t', str(segundos)]

        if segundo_inicio and segundo_inicio > 0 and segundos:
            seg_txt = f's{segundo_inicio}→s{segundo_inicio + segundos}'
        elif segundo_inicio and segundo_inicio > 0:
            seg_txt = f'desde s{segundo_inicio}'
        elif segundos:
            seg_txt = f'primeros {segundos}s'
        else:
            seg_txt = 'completo'

        step(f'Extrayendo frames ({seg_txt})…', 20)
        subprocess.run(
            ['ffmpeg', '-y'] + args_tiempo + ['-i', ruta_video,
             '-vf', 'fps=5,scale=640:-2',
             os.path.join(dir_frames, 'out%d.png')],
            capture_output=True
        )
        a.frames, i = [], 1
        while True:
            p = os.path.join(dir_frames, f'out{i}.png')
            if not os.path.exists(p): break
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: break
            a.frames.append(img)
            i += 1
        a.total_frames = len(a.frames)
        a.frame_color  = cv2.imread(os.path.join(dir_frames, 'out1.png')) if a.total_frames else None
        step(f'{a.total_frames} frames cargados', 30)

        # Guardar miniatura antes de que se borren los frames
        thumb_path = None
        if task_id and a.frame_color is not None:
            thumb_path = os.path.join(DIR_TRABAJO, 'thumbs', f'{task_id}.jpg')
            cv2.imwrite(thumb_path, a.frame_color, [cv2.IMWRITE_JPEG_QUALITY, 85])

        step('Estructura I/P/B…', 36)
        a.analizar_frames_ipb()
        step('Nitidez…', 43)
        a.analizar_nitidez()
        step('Análisis FFT…', 50)
        a.analizar_fft()
        step('Flujo óptico…', 57)
        a.analizar_flujo_optico()
        step('Fotometría…', 63)
        a.analizar_fotometria()
        step('Aberración cromática…', 68)
        a.analizar_aberracion_cromatica()
        step('Compresión DCT…', 73)
        a.analizar_compresion()
        step('Ruido…', 78)
        a.analizar_ruido()
        step('Correlación RGB…', 83)
        a.analizar_correlacion_rgb()
        step('Modelo ML + SHAP…', 90)
        a.predecir_con_modelo()

        # ── Análisis facial complementario (si hay cara) ──────
        step('Análisis facial…', 92)
        prob_facial    = None
        score_facial   = None
        cara_detectada = False
        try:
            from analisis_facial import calcular_features_faciales
            frames_color_facial = []
            for i in range(1, min(a.total_frames + 1, 31)):
                p = os.path.join(dir_frames, f'out{i}.png')
                if os.path.exists(p):
                    img = cv2.imread(p)
                    if img is not None:
                        frames_color_facial.append(img)

            feats_faciales = calcular_features_faciales(
                frames_color_facial, fps_video=5
            )
            cara_detectada = feats_faciales.get('face_detectada', 0) >= 1.0

            if cara_detectada and _modelo_facial is not None:
                X_f = np.array([[feats_faciales.get(f, 0.0)
                                 for f in _fnames_facial]])
                prob_facial  = float(_modelo_facial.predict_proba(X_f)[0, 1])
                score_facial = round(prob_facial * 100, 1)
                print(f"  [+] Score facial: {score_facial}%")
            elif cara_detectada and _modelo_facial is None:
                print("  [!] Cara detectada pero modelo facial no disponible")
        except Exception as e:
            print(f"  [!] Análisis facial falló: {e}")
            traceback.print_exc()

        # ── Análisis de contexto con Gemini ───────────────────
        step('Análisis de contexto visual (Gemini)…', 95)
        gemini_resultado = analizar_contexto_gemini(dir_frames, a.total_frames)

        step('Generando resultado…', 98)

        prob     = a.informe.get('ml_prob_ia', 0.0)
        prob_raw = a.informe.get('ml_prob_ia_raw', prob)
        origen   = a.informe.get('origen_detectado', 'desconocido')
        factor   = FACTORES_ORIGEN.get(origen, 1.0)
        score    = round(prob * 100, 1)

        if score >= 65:   veredicto, nivel = "SINTÉTICO / MANIPULADO", "danger"
        elif score >= 35: veredicto, nivel = "SOSPECHOSO",             "warning"
        else:             veredicto, nivel = "PROBABLEMENTE AUTÉNTICO","safe"

        # Veredicto facial análogo (solo si hay cara)
        veredicto_facial = nivel_facial = None
        if score_facial is not None:
            if score_facial >= 65:
                veredicto_facial, nivel_facial = "CARA SINTÉTICA", "danger"
            elif score_facial >= 35:
                veredicto_facial, nivel_facial = "CARA SOSPECHOSA", "warning"
            else:
                veredicto_facial, nivel_facial = "CARA AUTÉNTICA", "safe"

        shap_items = [
            {
                'feature':     it['feature'],
                'valor':       round(float(it['valor']), 4),
                'shap':        round(float(it['shap']),  4),
                'hacia_ia':    it['shap'] > 0,
                'descripcion': DESCRIPCIONES.get(it['feature'], ''),
            }
            for it in a.informe.get('shap_explicacion', [])
        ]

        emit('result',
             score=score,
             prob_raw=round(prob_raw * 100, 1),
             factor=factor,
             origen=origen,
             veredicto=veredicto,
             nivel=nivel,
             shap_items=shap_items,
             segundos_analizados=segundos,
             segundo_inicio=segundo_inicio,
             seg_txt=seg_txt,
             has_thumb=thumb_path is not None,
             gemini=gemini_resultado,
             modelo_usado=tipo_modelo,
             origen_usado=origen_final,
             cara_detectada=cara_detectada,
             score_facial=score_facial,
             veredicto_facial=veredicto_facial,
             nivel_facial=nivel_facial,
             raw={k: (round(float(v), 4) if isinstance(v, float) else v)
                  for k, v in a.informe.items()
                  if isinstance(v, (int, float)) and not k.startswith('origen_')},
        )

    except Exception as e:
        traceback.print_exc()
        emit('error', msg=str(e))
    finally:
        shutil.rmtree(dir_frames, ignore_errors=True)


def descargar_url(url, dest_dir, q, segundo_inicio=0, segundos=None):
    inicio  = segundo_inicio or 0
    duracion = segundos or 30
    fin     = inicio + duracion
    if inicio > 0:
        label = f'Descargando vídeo (s{inicio}→s{fin})…'
    else:
        label = f'Descargando vídeo (primeros {duracion}s)…'
    q.put({'type': 'progress', 'label': label, 'pct': 3})
    import yt_dlp
    opts = {
        'outtmpl':            os.path.join(dest_dir, '%(id)s.%(ext)s'),
        'format':             'best[ext=mp4]/best',
        'writeinfojson':      True,
        'quiet':              True,
        'no_warnings':        True,
        'download_ranges':    yt_dlp.utils.download_range_func(None, [(inicio, fin)]),
        'force_keyframes_at_cuts': True,
        'merge_output_format': 'mp4',
        'cookiesfrombrowser': ('firefox',),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


# ════════════════════════════════════════════════════════════════
#  RUTAS
# ════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('static', 'index3.html')


@app.route('/video/<task_id>')
def serve_video(task_id):
    """Sirve el vídeo con soporte de range requests para el reproductor."""
    ruta = videos.get(task_id)
    if not ruta or not os.path.exists(ruta):
        return 'Not found', 404

    file_size = os.path.getsize(ruta)
    range_header = request.headers.get('Range', None)

    if range_header:
        # Parsear el rango pedido por el navegador
        byte_start, byte_end = 0, None
        match = __import__('re').search(r'(\d+)-(\d*)', range_header)
        if match:
            byte_start = int(match.group(1))
            byte_end   = int(match.group(2)) if match.group(2) else file_size - 1
        byte_end = min(byte_end, file_size - 1)
        length   = byte_end - byte_start + 1

        with open(ruta, 'rb') as f:
            f.seek(byte_start)
            data = f.read(length)

        resp = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
        resp.headers['Content-Range']  = f'bytes {byte_start}-{byte_end}/{file_size}'
        resp.headers['Accept-Ranges']  = 'bytes'
        resp.headers['Content-Length'] = str(length)
        return resp
    else:
        resp = send_file(ruta, mimetype='video/mp4', conditional=True)
        resp.headers['Accept-Ranges'] = 'bytes'
        return resp


@app.route('/thumb/<task_id>')
def serve_thumb(task_id):
    """Sirve la miniatura del primer frame."""
    path = os.path.join(DIR_TRABAJO, 'thumbs', f'{task_id}.jpg')
    if not os.path.exists(path):
        return 'Not found', 404
    return send_file(path, mimetype='image/jpeg')


@app.route('/duracion', methods=['POST'])
def get_duracion():
    """Devuelve la duración de un vídeo ya guardado."""
    task_id = request.json.get('task_id')
    ruta = videos.get(task_id)
    if not ruta:
        return jsonify({'duracion': 0})
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
           '-of', 'json', ruta]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        dur = float(json.loads(res.stdout)['format']['duration'])
    except Exception:
        dur = 0
    return jsonify({'duracion': round(dur, 1)})


@app.route('/analyze', methods=['POST'])
def analyze():
    task_id    = uuid.uuid4().hex
    q          = Queue()
    tareas[task_id] = q
    upload_dir = os.path.join(DIR_TRABAJO, 'uploads')
    dir_frames = os.path.join(DIR_TRABAJO, f'frames_{task_id}')

    # Leer segundos elegidos por el usuario (0 = completo)
    try:
        segundos = int(request.form.get('segundos', 0)) or None
    except Exception:
        segundos = None

    # Segundo de inicio para fragmento personalizado (0 = desde el principio)
    try:
        segundo_inicio = int(request.form.get('segundo_inicio', 0)) or 0
    except Exception:
        segundo_inicio = 0

    # Origen seleccionado manualmente por el usuario ('auto' = detección automática)
    origen_manual = request.form.get('origen_manual', 'auto')

    # Leer archivo/URL dentro del contexto de Flask
    ruta_guardada = None
    url_pegada    = None

    if 'file' in request.files and request.files['file'].filename:
        f = request.files['file']
        ruta_guardada = os.path.join(upload_dir, task_id + Path(f.filename).suffix)
        f.save(ruta_guardada)
        videos[task_id] = ruta_guardada   # registrar para /video/<task_id>
    elif request.form.get('url'):
        url_pegada = request.form['url'].strip()

    def run():
        try:
            if ruta_guardada:
                ejecutar_analisis(ruta_guardada, dir_frames, q,
                                  segundos=segundos, task_id=task_id,
                                  origen_manual=origen_manual,
                                  segundo_inicio=segundo_inicio)
            elif url_pegada:
                src_dir = os.path.join(upload_dir, task_id)
                os.makedirs(src_dir, exist_ok=True)
                try:
                    ruta = descargar_url(url_pegada, src_dir, q,
                                         segundo_inicio=segundo_inicio,
                                         segundos=segundos)
                    videos[task_id] = ruta
                    ejecutar_analisis(ruta, dir_frames, q,
                                      segundos=segundos, task_id=task_id,
                                      origen_manual=origen_manual,
                                      segundo_inicio=0)  # ya recortado por yt-dlp
                except Exception as e:
                    q.put({'type': 'error', 'msg': f'Error descargando: {e}'})
            else:
                q.put({'type': 'error', 'msg': 'Sin archivo ni URL.'})
        finally:
            q.put({'type': 'done'})

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/stream/<task_id>')
def stream(task_id):
    def generate():
        if task_id not in tareas:
            yield f"data: {json.dumps({'type':'error','msg':'Tarea no encontrada'})}\n\n"
            return
        q = tareas[task_id]
        while True:
            try:
                msg = q.get(timeout=180)
                yield f"data: {json.dumps(msg, default=str)}\n\n"
                if msg.get('type') in ('result', 'error', 'done'):
                    tareas.pop(task_id, None)
                    break
            except Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print("=" * 55)
    print("  Analizador Forense  →  http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, threaded=True, port=5000)