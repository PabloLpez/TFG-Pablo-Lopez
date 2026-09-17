"""
main.py  —  Analizador forense de vídeo con detector de IA
-----------------------------------------------------------
Optimización clave: los frames se leen UNA vez del disco y se
reutilizan en todos los análisis (4× menos IO que la versión anterior).
Las heurísticas del reporte final están en una tabla declarativa
en vez de una cascada de llamadas.

Si el vídeo fue descargado con yt-dlp --write-info-json, se lee
el .info.json para detectar el origen (twitter, instagram, etc.)
y añadir el one-hot al vector de features del modelo.
"""

import subprocess, json, os, glob, pickle
import cv2
import numpy as np
from scipy.stats import shapiro
from scipy import stats
import shutil


# ============================================================
#  CONFIGURACIÓN
# ============================================================
FPS_EXTRACCION = 5
ANCHO_MAX      = 640
DIR_FRAMES     = "frames_tmp"

UMBRAL_BLUR, UMBRAL_FFT, UMBRAL_FLUJO = 50, 1.5, 3.0
UMBRAL_LUZ,  UMBRAL_CA,  UMBRAL_DCT   = 25, 0.05, 4.0
UMBRAL_SHAPIRO = 0.05

ORIGENES = ['movil_directo', 'instagram', 'twitter', 'whatsapp', 'youtube', 'desconocido']

FACTORES_ORIGEN = {
    'instagram': 0.85,
    'twitter':   0.88,
    'whatsapp':  0.90,
    'youtube':   0.92,
}

PESOS = {
    'meta_num_alertas': 12, 'fft_ratio_std': 12, 'flujo_salto_maximo': 9,
    'fft_energia_media': 8, 'ipb_pct_p': 6, 'ruido_asimetria': 4,
    'ruido_curtosis': 4, 'ipb_pct_i': 4, 'ipb_gop_std': 4,
    'ipb_gop_medio': 3, 'ipb_num_alertas': 3, 'ipb_pct_b': 3,
    'fft_ratio_alta_baja': 3, 'nitidez_std': 3, 'flujo_medio': 3,
    'ipb_ratio_tamano_ip': 2, 'brillo_std': 2, 'ca_desfase_std': 2,
    'nitidez_outlier_pct': 2, 'brillo_tendencia': 2,
    'ca_desfase_medio': 1, 'nitidez_media': 1, 'dct_ratio': 1,
    'ruido_std': 1, 'luz_std': 1,
}

# Tabla heurística: (clave_peso, condición(informe), descripción(informe))
def _g(d, k, default=0): return d.get(k, default)

HEURISTICAS = [
    ('ipb_num_alertas',     lambda i: _g(i, 'ipb_num_alertas') >= 2,
                            lambda i: f"IPB: {_g(i, 'ipb_num_alertas')} alertas de compresión"),
    ('ipb_pct_p',           lambda i: _g(i, 'ipb_pct_p', 100) < 30,
                            lambda i: f"Pocos P-frames ({_g(i, 'ipb_pct_p'):.1f}%)"),
    ('ipb_pct_b',           lambda i: _g(i, 'ipb_pct_b', 100) == 0,
                            "Sin B-frames (IA o reexportación)"),
    ('ipb_pct_i',           lambda i: _g(i, 'ipb_pct_i') > 20,
                            lambda i: f"Exceso I-frames ({_g(i, 'ipb_pct_i'):.1f}%)"),
    ('ipb_gop_std',         lambda i: _g(i, 'ipb_gop_std', 1) < 1.0,
                            "GOP robótico (cadencia perfecta)"),
    ('ipb_ratio_tamano_ip', lambda i: 0 < _g(i, 'ipb_ratio_tamano_ip', 99) < 1.5,
                            "Ratio tamaño I/P anómala (<1.5)"),
    ('ipb_gop_medio',       lambda i: 0 < _g(i, 'ipb_gop_medio') < 5,
                            "GOP muy corto (posible codificación IA)"),

    ('fft_ratio_std',       lambda i: _g(i, 'fft_ratio_std') > 0.3,
                            "FFT: variación de ratio alta/baja inusual"),
    ('fft_ratio_alta_baja', lambda i: _g(i, 'fft_ratio_alta_baja') > UMBRAL_FFT,
                            "FFT: exceso de altas frecuencias"),
    ('fft_energia_media',   lambda i: _g(i, 'fft_energia_media') > 80,
                            "FFT: energía media elevada"),

    ('ruido_std',           lambda i: _g(i, 'ruido_shapiro_p') > UMBRAL_SHAPIRO,
                            "Ruido gaussiano perfecto (sintético)"),
    ('ruido_asimetria',     lambda i: abs(_g(i, 'ruido_asimetria', 99)) < 0.15,
                            "Asimetría de ruido casi nula (IA)"),
    ('ruido_curtosis',      lambda i: abs(_g(i, 'ruido_curtosis', 99)) < 0.15,
                            "Curtosis de ruido casi nula (IA)"),

    ('meta_num_alertas',    lambda i: _g(i, 'meta_num_alertas') >= 2,
                            lambda i: f"Metadatos: {_g(i, 'meta_num_alertas')} inconsistencias"),

    ('nitidez_std',         lambda i: _g(i, 'nitidez_std') > UMBRAL_BLUR,
                            "Variación de nitidez inusual entre frames"),
    ('nitidez_outlier_pct', lambda i: _g(i, 'nitidez_outlier_pct') > 0.1,
                            ">10% frames con nitidez anómala"),

    ('flujo_medio',         lambda i: _g(i, 'flujo_medio') > 8,
                            "Movimiento medio excesivo"),
    ('flujo_salto_maximo',  lambda i: _g(i, 'flujo_salto_maximo') > UMBRAL_FLUJO,
                            "Salto temporal brusco en flujo óptico"),

    ('brillo_std',          lambda i: _g(i, 'brillo_std') > 15,
                            "Variación de brillo inusual"),
    ('ca_desfase_std',      lambda i: _g(i, 'ca_desfase_std', 1) < UMBRAL_CA,
                            "Lente sintética (aberración cromática uniforme)"),
    ('dct_ratio',           lambda i: _g(i, 'dct_ratio') > UMBRAL_DCT,
                            "Patrón DCT anómalo (recompresión múltiple)"),
]


# ============================================================
#  CLASE
# ============================================================

class AnalizadorVideo:
    def __init__(self, ruta):
        self.ruta              = ruta
        self.informe           = {}
        self.frames            = []     # cache de frames grayscale
        self.frame_color       = None   # cache del primer frame en color
        self.total_frames      = 0
        self.segundos_analizar = None
        os.makedirs(DIR_FRAMES, exist_ok=True)

    # ─── duración / segmento ───────────────────────────────
    def obtener_duracion(self):
        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
               '-of', 'json', self.ruta]
        try:
            return float(json.loads(subprocess.run(
                cmd, capture_output=True, text=True).stdout)['format']['duration'])
        except Exception:
            print("[!] No se pudo obtener la duración del vídeo.")
            return 0.0

    def elegir_segmento(self):
        d = self.obtener_duracion()
        if d <= 0:
            return
        print(f"\n[-] Duración: {int(d//60)}m {int(d%60):02d}s ({d:.1f} s)")

        if d <= 30:
            print("[-] El vídeo dura ≤30 s → se procesará completo.")
            return

        if d > 60:
            opts = {'1': ('30 segundos', 30), '2': ('60 segundos', 60),
                    '3': ('Vídeo completo', None)}
            print("\n¿Cuánto analizar?\n  [1] 30 s\n  [2] 60 s\n  [3] Completo")
        else:
            opts = {'1': ('30 segundos', 30), '2': ('Vídeo completo', None)}
            print("\n¿Cuánto analizar?\n  [1] 30 s\n  [2] Completo")

        while True:
            e = input("\nElige una opción: ").strip()
            if e in opts:
                nombre, valor = opts[e]
                self.segundos_analizar = valor
                print(f"[-] Seleccionado: {nombre}")
                return
            print(f"    [!] Opción no válida: {', '.join(opts)}")

    # ─── origen (yt-dlp) ───────────────────────────────────
    def detectar_origen(self):
        json_p = os.path.splitext(self.ruta)[0] + '.info.json'
        origen = 'movil_directo'
        if os.path.exists(json_p):
            try:
                ext = json.load(open(json_p, encoding='utf-8')).get('extractor', '').lower()
                origen = next((k for k in ('instagram', 'twitter', 'whatsapp', 'youtube')
                               if k in ext), 'twitter' if 'x.com' in ext else 'desconocido')
            except Exception:
                origen = 'desconocido'
            print(f"\n[-] Origen detectado (yt-dlp): {origen.upper()}")
        else:
            print(f"\n[-] Sin .info.json → origen asumido: {origen.upper()}")

        self.informe['origen_detectado'] = origen
        for o in ORIGENES:
            self.informe[f'origen_{o}'] = int(origen == o)
        return origen

    # ─── modelo ML ─────────────────────────────────────────
    def cargar_modelo(self, ruta_modelo='modelo.pkl'):
        with open(ruta_modelo, 'rb') as f:
            p = pickle.load(f)
        self.ml_modelo   = p.get('modelo') or p.get('model')
        self.ml_features = p['feature_names']
        print(f"[-] Modelo ML cargado: {p.get('best_model_name', '?').upper()}")

    def predecir_con_modelo(self):
        vec = [float(self.informe.get(f, 0.0)) for f in self.ml_features]
        falt = [f for f in self.ml_features if f not in self.informe]
        if falt:
            print(f"    [!] Features no calculadas (→ 0): {falt}")
        X    = np.array([vec])
        prob = float(self.ml_modelo.predict_proba(X)[0][1])
        ver  = "IA / SINTÉTICO" if prob > 0.5 else "AUTÉNTICO"
        self.informe['ml_prob_ia'], self.informe['ml_veredicto'] = prob, ver
        print(f"\n[ML] Probabilidad IA: {prob:.1%}  →  {ver}")

        # ── Explicación SHAP ─────────────────────────────────
        # Descripciones en lenguaje natural para el usuario final
        DESCRIPCIONES = {
            'ruido_corr_temporal':
                "Correlación del ruido entre frames consecutivos. Una cámara real deja "
                "una huella persistente en el sensor que se repite de frame a frame. "
                "Si esta correlación es baja, el vídeo no tiene esa huella física.",
            'ruido_planitud_espectral':
                "Planitud del espectro de ruido. El ruido de un sensor real tiene más "
                "energía en frecuencias bajas (patrón 1/f). El ruido sintético es "
                "completamente uniforme en todas las frecuencias.",
            'ruido_asimetria':
                "Asimetría de la distribución del ruido. El ruido de sensor real no es "
                "perfectamente simétrico. Una asimetría casi nula indica que el ruido "
                "fue generado matemáticamente, no capturado por hardware.",
            'ruido_curtosis':
                "Curtosis del ruido. Las cámaras reales producen ruido con colas "
                "más pesadas de lo normal. Una curtosis casi nula es característica "
                "de ruido sintético perfectamente gaussiano.",
            'ruido_corr_canales':
                "Correlación del ruido entre canales de color. El proceso de "
                "interpolación de color de las cámaras (demosaicing) genera una "
                "correlación natural entre canales R y G. La IA no pasa por ese proceso.",
            'ruido_std_temporal':
                "Variación del nivel de ruido a lo largo del vídeo. El ruido de una "
                "cámara real varía con la exposición y el contenido. Un nivel de ruido "
                "perfectamente constante entre frames indica síntesis artificial.",
            'ruido_shapiro_p':
                "Test de gaussianidad del ruido. El ruido real de sensor no es "
                "perfectamente gaussiano. Si pasa el test con nota, el ruido es "
                "demasiado perfecto para ser real.",
            'ruido_std':
                "Nivel medio de ruido. Relacionado con la distribución estadística "
                "del residuo de ruido extraído de la imagen.",
            'meta_num_alertas':
                "Inconsistencias en los metadatos del archivo. Una grabación real "
                "contiene fabricante, modelo de cámara y fecha de grabación. "
                "Su ausencia o incoherencia puede indicar generación artificial.",
            'fft_ratio_std':
                "Variación de la distribución de frecuencias entre frames. Los vídeos "
                "reales cambian de escena a escena de forma natural. Una variación "
                "inusualmente alta puede indicar inconsistencia generada por IA.",
            'fft_energia_media':
                "Energía media del espectro de frecuencias. Valores muy altos pueden "
                "indicar sharpening artificial aplicado por un generador de IA o "
                "por la recodificación agresiva de plataformas sociales.",
            'fft_ratio_alta_baja':
                "Proporción de energía en altas vs bajas frecuencias. Las imágenes "
                "naturales tienen más energía en bajas frecuencias. Un exceso de "
                "altas frecuencias indica procesado artificial.",
            'ipb_pct_b':
                "Porcentaje de B-frames en el vídeo. Los codificadores reales usan "
                "B-frames para comprimir mejor. Su ausencia total es típica de vídeos "
                "generados por IA o reexportados frame a frame.",
            'ipb_pct_i':
                "Porcentaje de I-frames (imágenes completas). En vídeos normales son "
                "menos del 5%. Un porcentaje alto indica que el codificador no "
                "aprovechó la redundancia temporal, como ocurre en vídeos de IA.",
            'ipb_pct_p':
                "Porcentaje de P-frames (diferencias con el frame anterior). "
                "Deben ser el tipo dominante. Un porcentaje anormalmente bajo indica "
                "que la codificación no siguió el patrón natural.",
            'ipb_gop_std':
                "Variación en el tamaño de los grupos de frames (GOP). Un codificador "
                "real adapta el GOP al contenido. Un GOP perfectamente regular "
                "indica una codificación mecánica sin adaptación.",
            'ipb_gop_medio':
                "Tamaño medio del grupo de frames. Un GOP muy corto indica que el "
                "codificador no está aprovechando la redundancia temporal entre frames, "
                "lo que es inusual en grabaciones reales.",
            'ipb_ratio_tamano_ip':
                "Diferencia de tamaño entre I-frames y P-frames. Los I-frames deben "
                "ser significativamente más pesados. Una ratio baja indica que los "
                "frames no contienen la información esperada.",
            'ipb_num_alertas':
                "Número total de anomalías detectadas en la estructura de compresión "
                "del vídeo. Agrupa los indicadores anteriores de I/P/B-frames.",
            'gradiente_curtosis':
                "Curtosis de la distribución de bordes de la imagen. Las imágenes "
                "reales tienen bordes muy pronunciados en pocos puntos y suavidad en "
                "el resto. La IA tiende a distribuir los bordes de forma más uniforme.",
            'gradiente_entropia':
                "Uniformidad de las orientaciones de los bordes. Las imágenes reales "
                "tienen direcciones de borde preferentes. Una distribución demasiado "
                "uniforme de orientaciones indica generación artificial.",
            'flujo_suavidad':
                "Suavidad del movimiento entre frames. Una cámara real tiene "
                "pequeñas imperfecciones e inercia. Un movimiento anormalmente suave "
                "y predecible es típico de vídeos generados.",
            'flujo_salto_maximo':
                "Salto brusco máximo en el movimiento entre frames. Saltos muy altos "
                "sin un corte de plano que los justifique indican discontinuidades "
                "temporales artificiales.",
            'flujo_medio':
                "Magnitud media del movimiento a lo largo del vídeo. Valores "
                "inusualmente altos pueden indicar que el generador no controló "
                "correctamente la coherencia temporal.",
            'flujo_std':
                "Variación del movimiento a lo largo del vídeo.",
            'flujo_num_glitches':
                "Número de anomalías estadísticas en el movimiento entre frames.",
            'nitidez_std':
                "Variación de nitidez entre frames. Una cámara real tiene variaciones "
                "naturales por el enfoque automático y el movimiento. Una variación "
                "muy alta o muy baja puede indicar síntesis.",
            'nitidez_outlier_pct':
                "Porcentaje de frames con nitidez muy diferente al resto. Un "
                "porcentaje alto indica que el generador no mantuvo coherencia "
                "de foco a lo largo del vídeo.",
            'nitidez_media':
                "Nivel medio de nitidez de los frames del vídeo.",
            'brillo_std':
                "Variación del brillo entre frames. Cambios bruscos de exposición "
                "que no corresponden a cambios reales en la escena pueden indicar "
                "que la iluminación fue generada artificialmente.",
            'brillo_tendencia':
                "Tendencia lineal del brillo a lo largo del vídeo. Una tendencia "
                "muy pronunciada puede indicar un fade artificial aplicado por "
                "el generador.",
            'luz_std':
                "Variación de la dirección de iluminación entre frames. Cambios "
                "bruscos en la dirección de la luz sin justificación física "
                "indican inconsistencias de una escena generada.",
            'ca_desfase_std':
                "Uniformidad de la aberración cromática entre zonas de la imagen. "
                "Las lentes reales tienen más aberración en los bordes que en el "
                "centro. Una aberración perfectamente uniforme indica lente sintética.",
            'ca_desfase_medio':
                "Nivel medio de aberración cromática. El desplazamiento entre "
                "los canales de color que produce cualquier lente óptica real.",
            'dct_ratio':
                "Patrón de compresión en bloques. Un ratio alto indica que el "
                "vídeo ha sido comprimido y recomprimido múltiples veces, "
                "dejando artefactos en la cuadrícula de compresión.",
            'corr_rg':
                "Correlación entre los canales rojo y verde. Valores muy altos "
                "indican que los canales son casi idénticos, lo que puede sugerir "
                "que el vídeo fue generado sin variación cromática real.",
            'corr_rb':
                "Correlación entre los canales rojo y azul.",
            'corr_gb':
                "Correlación entre los canales verde y azul.",
        }

        try:
            import shap
            clf = self.ml_modelo.named_steps['clf']

            # TreeExplainer funciona con GBM, RF y XGBoost
            # Para GBM de scikit-learn hay que pasar check_additivity=False
            explicador = shap.TreeExplainer(clf)
            shap_vals  = explicador.shap_values(X, check_additivity=False)

            # shap_values puede devolver lista (clasificación binaria) o array
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_vals = np.array(shap_vals).reshape(1, -1)

            contribuciones = sorted(
                zip(self.ml_features, shap_vals[0]),
                key=lambda x: abs(x[1]), reverse=True
            )

            print(f"\n{'='*70}")
            print(f" POR QUÉ EL MODELO HA DADO {prob:.1%} ".center(70, "="))
            print(f"{'='*70}")
            print("  (+) empuja hacia IA   |   (-) empuja hacia AUTÉNTICO\n")

            for feat, val in contribuciones[:10]:
                if abs(val) < 0.001:
                    continue
                dir_txt  = "→ IA   " if val > 0 else "→ REAL "
                val_real = self.informe.get(feat, 0.0)
                desc     = DESCRIPCIONES.get(feat, "")

                print(f"  {'(+)' if val > 0 else '(-)'} {feat}")
                print(f"      Valor medido : {val_real:.4f}   "
                      f"Influencia: {val:+.4f}  {dir_txt}")
                if desc:
                    # Imprimir descripción en líneas de máx 65 caracteres
                    palabras, linea = desc.split(), ""
                    for p in palabras:
                        if len(linea) + len(p) + 1 > 65:
                            print(f"      {linea}")
                            linea = p
                        else:
                            linea = (linea + " " + p).strip()
                    if linea:
                        print(f"      {linea}")
                print()

            self.informe['shap_explicacion'] = [
                {'feature': f, 'valor': float(self.informe.get(f, 0.0)),
                 'shap': float(v)}
                for f, v in contribuciones[:10]
            ]

        except ImportError:
            print("    [i] instala 'shap' para ver la explicación del modelo: pip install shap")
        except Exception as e:
            print(f"    [!] SHAP no disponible: {e}")

        return prob

    # ─── frames ────────────────────────────────────────────
    def _ruta_frame(self, i):
        return os.path.join(DIR_FRAMES, f"out{i}.png")

    def limpiar_frames(self):
        for f in glob.glob(f'{DIR_FRAMES}/out*.png'):
            os.remove(f)
        print("[-] Frames temporales eliminados.")

    def extraer_frames(self):
        args_t = ['-t', str(self.segundos_analizar)] if self.segundos_analizar else []
        msg    = (f"primeros {self.segundos_analizar} s" if self.segundos_analizar
                  else "vídeo completo")
        print(f"[-] Extrayendo frames ({msg})...")

        subprocess.run(
            ['ffmpeg', '-y'] + args_t + ['-i', self.ruta,
             '-vf', f'fps={FPS_EXTRACCION},scale={ANCHO_MAX}:-2',
             os.path.join(DIR_FRAMES, 'out%d.png')],
            capture_output=True
        )

        # Cargar a memoria de una vez (4× menos IO en los análisis siguientes)
        self.frames, i = [], 1
        while True:
            img = cv2.imread(self._ruta_frame(i), cv2.IMREAD_GRAYSCALE)
            if img is None:
                break
            self.frames.append(img)
            i += 1
        self.total_frames = len(self.frames)
        self.frame_color  = cv2.imread(self._ruta_frame(1)) if self.total_frames else None
        print(f"[-] Frames extraídos y cargados a memoria: {self.total_frames} "
              f"({FPS_EXTRACCION} fps, máx {ANCHO_MAX}px)")

    # ─── metadatos ─────────────────────────────────────────
    def analizar_metadatos(self):
        ruta_exiftool = os.environ.get('EXIFTOOL_PATH') or shutil.which('exiftool') or 'exiftool'
        res = subprocess.run([ruta_exiftool, '-j', self.ruta],
                             capture_output=True, text=True)
        if res.stdout.strip():
            self.informe['_metadatos_raw'] = json.loads(res.stdout)
            print("[-] Metadatos leídos.")
        else:
            print("[!] ExifTool sin datos.")
            self.informe['_metadatos_raw'] = [{}]

    def analizar_metadatos_forenses(self):
        print("\n[-] Análisis forense de metadatos...")
        meta = (self.informe.get('_metadatos_raw') or [{}])[0]
        a = 0

        fc, fm = meta.get('CreateDate'), meta.get('ModifyDate')
        if fc and fm and fc > fm:
            a += 1
            print("    [!] Paradoja temporal en timestamps.")

        fab = meta.get('Make', '').lower()
        if fab in {'unknown', 'adobe', 'synthesia', 'runway', 'generic', ''} or 'ai' in fab:
            a += 1
            print(f"    [!] Fabricante sospechoso: '{fab}'")

        ausentes = [c for c in ('Make', 'Model', 'CreateDate', 'Software') if not meta.get(c)]
        if len(ausentes) >= 3:
            a += 1
            print(f"    [!] Faltan campos críticos: {ausentes}")

        sw = meta.get('Software', '').lower()
        if any(e in sw for e in ('adobe', 'premiere', 'after effects', 'davinci', 'final cut')):
            print(f"    [i] Software de edición: {sw}")

        self.informe['meta_num_alertas'] = a
        print(f"[-] Metadatos: {a} alerta(s).")

    # ─── I/P/B ─────────────────────────────────────────────
    def analizar_frames_ipb(self):
        print("\n[-] Análisis I/P/B...")

        ft = ['-read_intervals', f'%+{self.segundos_analizar}'] if self.segundos_analizar else []
        cmd = (['ffprobe', '-v', 'quiet', '-select_streams', 'v:0'] + ft +
               ['-show_frames', '-show_entries',
                'frame=pict_type,pkt_size,best_effort_timestamp_time',
                '-of', 'json', self.ruta])
        res = subprocess.run(cmd, capture_output=True, text=True)
        frames = json.loads(res.stdout).get('frames', []) if res.stdout.strip() else []
        if not frames:
            print("[!] ffprobe sin datos.")
            for k in ('ipb_pct_i', 'ipb_pct_p', 'ipb_pct_b', 'ipb_gop_medio',
                      'ipb_gop_std', 'ipb_ratio_tamano_ip', 'ipb_num_alertas'):
                self.informe[k] = 0.0
            return

        conteos = {'I': 0, 'P': 0, 'B': 0}
        tamanos = {'I': [], 'P': [], 'B': []}
        gops, gop = [], 0
        for fr in frames:
            t, sz = fr.get('pict_type', '?'), int(fr.get('pkt_size', 0))
            if t in conteos:
                conteos[t] += 1
                tamanos[t].append(sz)
            if t == 'I':
                if gop > 0:
                    gops.append(gop)
                gop = 1
            else:
                gop += 1
        if gop > 0:
            gops.append(gop)

        total = sum(conteos.values()) or 1
        pct   = {k: conteos[k] / total * 100 for k in 'IPB'}
        gm    = float(np.mean(gops)) if gops else 0.0
        gs    = float(np.std(gops))  if gops else 0.0
        mi    = np.mean(tamanos['I']) if tamanos['I'] else 0
        mp    = np.mean(tamanos['P']) if tamanos['P'] else 1
        rip   = float(mi / (mp + 1e-6))

        print(f"    I:{conteos['I']}({pct['I']:.1f}%) "
              f"P:{conteos['P']}({pct['P']:.1f}%) "
              f"B:{conteos['B']}({pct['B']:.1f}%) | "
              f"GOP:{gm:.1f}±{gs:.2f} | I/P:{rip:.2f}")

        alertas, msgs = 0, []
        if conteos['B'] == 0:                  alertas += 1; msgs.append("Sin B-frames")
        if pct['I'] > 20:                      alertas += 1; msgs.append(f"{pct['I']:.1f}% I-frames")
        if gs < 1.0 and gm > 0:                alertas += 1; msgs.append("GOP robótico")
        if 0 < rip < 1.5 and mi > 0 and mp > 0: alertas += 1; msgs.append(f"Ratio I/P={rip:.2f}")
        for m in msgs:
            print(f"    [!] {m}")

        self.informe.update({
            'ipb_pct_i': pct['I'], 'ipb_pct_p': pct['P'], 'ipb_pct_b': pct['B'],
            'ipb_gop_medio': gm, 'ipb_gop_std': gs,
            'ipb_ratio_tamano_ip': rip, 'ipb_num_alertas': alertas,
        })

        # Diferencia visual entre I-frames consecutivos (alerta extra si incoherentes)
        self._diff_iframes(frames)
        print(f"[-] I/P/B: {self.informe['ipb_num_alertas']} alerta(s).")

    def _diff_iframes(self, frames):
        ts_i = [float(f.get('best_effort_timestamp_time', 0) or 0)
                for f in frames if f.get('pict_type') == 'I'][:20]
        if len(ts_i) < 2:
            return

        diffs, prev = [], None
        for ts in ts_i:
            idx = max(1, int(ts * FPS_EXTRACCION) + 1)
            if idx > self.total_frames:
                continue
            img = self.frames[idx - 1]   # ◀ usa cache
            if prev is not None:
                if img.shape != prev.shape:
                    img = cv2.resize(img, (prev.shape[1], prev.shape[0]))
                diffs.append(np.mean(np.abs(img.astype(float) - prev.astype(float))))
            prev = img

        if diffs:
            med, std = float(np.mean(diffs)), float(np.std(diffs))
            if std > med * 0.8:
                self.informe['ipb_num_alertas'] += 1
                print(f"    [!] I-frames visualmente incoherentes (std={std:.1f} > 0.8·media).")

    # ─── nitidez ───────────────────────────────────────────
    def analizar_nitidez(self):
        print("\n[-] Análisis de nitidez...")
        if not self.frames:
            return
        vals = np.array([cv2.Laplacian(f, cv2.CV_64F).var() for f in self.frames])
        mu, sg = float(vals.mean()), float(vals.std())
        out    = float(np.mean(np.abs(vals - mu) > 2 * sg))
        self.informe.update({'nitidez_media': mu, 'nitidez_std': sg, 'nitidez_outlier_pct': out})
        print(f"    Media: {mu:.2f} | Std: {sg:.2f} | Outliers: {out:.1%}")
        if sg > UMBRAL_BLUR:
            print("    [!] Variación de nitidez inusual.")

    # ─── FFT ───────────────────────────────────────────────
    def analizar_fft(self):
        print("\n[-] Análisis FFT...")
        if not self.frames:
            return
        h, w = self.frames[0].shape
        m = np.zeros((h, w))
        cv2.circle(m, (w // 2, h // 2), min(h, w) // 8, 1, -1)
        inv = 1 - m

        energias, ratios = [], []
        for img in self.frames:
            mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1)
            energias.append(mag.mean())
            ratios.append((mag * inv).sum() / ((mag * m).sum() + 1e-6))

        e, r, rs = float(np.mean(energias)), float(np.mean(ratios)), float(np.std(ratios))
        self.informe.update({'fft_energia_media': e, 'fft_ratio_alta_baja': r, 'fft_ratio_std': rs})
        print(f"    Energía: {e:.2f} | Ratio: {r:.4f} | Std ratio: {rs:.4f}")
        if r > UMBRAL_FFT:
            print("    [!] Exceso de altas frecuencias.")

    # ─── flujo óptico ──────────────────────────────────────
    def analizar_flujo_optico(self):
        print("\n[-] Análisis de flujo óptico...")
        if len(self.frames) < 2:
            return

        h, w = self.frames[0].shape
        movs, saltos, glitches = [], [], 0
        prev = self.frames[0]
        for curr in self.frames[1:]:
            if curr.shape != prev.shape:
                curr = cv2.resize(curr, (w, h))
            flujo  = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
            movs.append(float(mag.mean()))
            if len(movs) > 1:
                d = abs(movs[-1] - movs[-2])
                saltos.append(d)
                if len(saltos) > 5:
                    arr = np.array(saltos[:-1])
                    if d > arr.mean() + 3 * arr.std():
                        glitches += 1
            prev = curr

        salto_max = float(max(saltos)) if saltos else 0.0
        self.informe.update({
            'flujo_medio': float(np.mean(movs)), 'flujo_std': float(np.std(movs)),
            'flujo_salto_maximo': salto_max, 'flujo_num_glitches': glitches,
        })
        print(f"    Movimiento medio: {self.informe['flujo_medio']:.2f} | "
              f"Salto máx: {salto_max:.2f} | Glitches: {glitches}")
        if salto_max > UMBRAL_FLUJO:
            print("    [!] Saltos temporales detectados.")

    # ─── fotometría ────────────────────────────────────────
    def analizar_fotometria(self):
        print("\n[-] Análisis fotométrico...")
        if not self.frames:
            return
        direcciones, brillos = [], []
        for img in self.frames:
            bl = cv2.GaussianBlur(img, (15, 15), 0)
            sx = cv2.Sobel(bl, cv2.CV_64F, 1, 0, ksize=5)
            sy = cv2.Sobel(bl, cv2.CV_64F, 0, 1, ksize=5)
            direcciones.append(float(np.arctan2(sy, sx).mean() * 180 / np.pi))
            brillos.append(float(img.mean()))

        ls, bs = float(np.std(direcciones)), float(np.std(brillos))
        bt = float(np.polyfit(range(len(brillos)), brillos, 1)[0])
        self.informe.update({'luz_std': ls, 'brillo_std': bs, 'brillo_tendencia': bt})
        print(f"    Std luz: {ls:.2f}° | Std brillo: {bs:.2f} | Tendencia: {bt:.4f}")
        if ls > UMBRAL_LUZ:
            print("    [!] Iluminación inconsistente.")

    # ─── aberración cromática ──────────────────────────────
    def analizar_aberracion_cromatica(self):
        print("\n[-] Análisis de aberración cromática...")
        if self.frame_color is None:
            return
        _, g, r = cv2.split(self.frame_color.astype(np.float32))
        h, w = self.frame_color.shape[:2]
        t = 100

        def parche(c, y, x):
            p = c[y - t // 2:y + t // 2, x - t // 2:x + t // 2]
            return p if p.shape == (t, t) else None

        desps = []
        for nombre, (y, x) in [('centro', (h//2, w//2)),
                                ('sup_izq', (t, t)),    ('sup_der', (t, w-t)),
                                ('inf_izq', (h-t, t)),  ('inf_der', (h-t, w-t))]:
            pg, pr = parche(g, y, x), parche(r, y, x)
            if pg is None or pr is None or pg.sum() == 0:
                continue
            shift, _ = cv2.phaseCorrelate(pg, pr)
            d = float(np.hypot(*shift))
            desps.append(d)
            print(f"    {nombre}: {d:.4f} px")

        if desps:
            cm, cs = float(np.mean(desps)), float(np.std(desps))
            self.informe.update({'ca_desfase_medio': cm, 'ca_desfase_std': cs})
            print(f"    Medio: {cm:.4f} | Std: {cs:.4f}")
            if cs < UMBRAL_CA:
                print("    [!] Lente sintética (aberración demasiado uniforme).")

    # ─── DCT ───────────────────────────────────────────────
    def analizar_compresion(self):
        print("\n[-] Análisis de compresión DCT...")
        if not self.frames:
            return
        img = self.frames[0]
        gx  = np.abs(np.diff(img.astype(float), axis=1))
        gy  = np.abs(np.diff(img.astype(float), axis=0))
        mgx, mgy = gx.mean(), gy.mean()

        def bloques(bs):
            h, w = img.shape
            c  = sum(1 for i in range(bs, h, bs) if i < gy.shape[0] and gy[i].mean() > mgy * 1.2)
            c += sum(1 for j in range(bs, w, bs) if j < gx.shape[1] and gx[:, j].mean() > mgx * 1.2)
            return c

        b8, b16 = bloques(8), bloques(16)
        ratio = float(b8 / (b16 + 1))
        self.informe['dct_ratio'] = ratio
        print(f"    Bloques 8×8: {b8} | 16×16: {b16} | Ratio: {ratio:.4f}")
        if ratio > UMBRAL_DCT:
            print("    [!] Patrón de compresión anómalo.")

    # ─── ruido ─────────────────────────────────────────────
    def analizar_ruido(self):
        print("\n[-] Análisis de ruido...")
        if not self.frames:
            return
        img   = self.frames[0]
        ruido = img.astype(float) - cv2.medianBlur(img, 5).astype(float)
        flat  = ruido.flatten()
        muestra = np.random.choice(flat, 5000, replace=False) if len(flat) > 5000 else flat
        _, p = shapiro(muestra)

        self.informe.update({
            'ruido_shapiro_p': float(p),
            'ruido_std':       float(ruido.std()),
            'ruido_asimetria': float(stats.skew(flat)),
            'ruido_curtosis':  float(stats.kurtosis(flat)),
        })
        print(f"    Shapiro p: {p:.6f} | Std: {self.informe['ruido_std']:.4f} | "
              f"Asim: {self.informe['ruido_asimetria']:.4f} | "
              f"Curt: {self.informe['ruido_curtosis']:.4f}")
        if p > UMBRAL_SHAPIRO:
            print("    [!] Ruido gaussiano perfecto. Típico de IA.")

    # ─── correlación RGB ───────────────────────────────────
    def analizar_correlacion_rgb(self):
        print("\n[-] Análisis correlación RGB...")
        if self.frame_color is None:
            return
        b, g, r = cv2.split(self.frame_color)
        rf, gf, bf = r.flatten(), g.flatten(), b.flatten()
        rg = float(np.corrcoef(rf, gf)[0, 1])
        rb = float(np.corrcoef(rf, bf)[0, 1])
        gb = float(np.corrcoef(gf, bf)[0, 1])
        self.informe.update({'corr_rg': rg, 'corr_rb': rb, 'corr_gb': gb})
        print(f"    R-G: {rg:.4f} | R-B: {rb:.4f} | G-B: {gb:.4f}")
        if rg > 0.95 and rb > 0.95:
            print("    [!] Canales RGB casi idénticos. Posible síntesis.")

    # ─── reporte final ─────────────────────────────────────
    def generar_reporte_final(self):
        print("\n" + "=" * 70)
        print(" REPORTE FORENSE FINAL ".center(70, "="))
        print("=" * 70)

        if self.segundos_analizar:
            print(f"[i] Análisis parcial: primeros {self.segundos_analizar} s.\n")
        print(f"[i] Origen: {self.informe.get('origen_detectado', '?').upper()}\n")

        # Aplicar tabla de heurísticas
        puntuacion, evidencias = 0, []
        for clave, cond, desc in HEURISTICAS:
            if cond(self.informe):
                pts = PESOS.get(clave, 1)
                puntuacion += pts
                texto = desc(self.informe) if callable(desc) else desc
                evidencias.append((pts, f"(+{pts:2d}) {texto}"))

        evidencias.sort(key=lambda x: -x[0])   # ordenar por peso descendente

        MAX = sum(PESOS.values())
        score = min(100, int(puntuacion / MAX * 100))

        if score >= 65:
            ver, tag = "SINTÉTICO / MANIPULADO  (Alta probabilidad)", "[!!!]"
        elif score >= 35:
            ver, tag = "SOSPECHOSO  (Revisión manual recomendada)", "[!!] "
        else:
            ver, tag = "PROBABLEMENTE AUTÉNTICO", "[ i ] "

        print(f"\n{tag} PUNTUACIÓN: {score}/100  ({puntuacion}/{MAX} pts)")
        print(f"{tag} VEREDICTO: {ver}\n")

        if evidencias:
            print("EVIDENCIAS DETECTADAS (por peso):")
            for _, ev in evidencias:
                print(f"  • {ev}")
        else:
            print("Sin anomalías significativas.")

        if 'ml_prob_ia' in self.informe:
            print(f"\n[ML] Probabilidad IA (modelo): {self.informe['ml_prob_ia']:.1%}")
        print("\n" + "=" * 70)

        self.informe.update({
            'puntuacion_heuristica': score,
            'veredicto_heuristico':  ver,
            'evidencias':            [e for _, e in evidencias],
            'segundos_analizados':   self.segundos_analizar,
        })

        with open('informe_forense.json', 'w', encoding='utf-8') as f:
            json.dump(
                {k: v for k, v in self.informe.items() if not k.startswith('_')},
                f, indent=2, ensure_ascii=False,
                default=lambda o: float(o) if isinstance(o, np.floating)
                         else int(o) if isinstance(o, np.integer) else str(o)
            )
        print("[✓] informe_forense.json guardado")


# ============================================================
#  EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" ANALIZADOR FORENSE DE VÍDEO — DETECTOR DE IA ".center(70))
    print("=" * 70)

    a = AnalizadorVideo("maquillaje.mp4")

    a.elegir_segmento()
    a.detectar_origen()
    a.cargar_modelo('modelo2.pkl')

    a.analizar_metadatos()
    a.analizar_metadatos_forenses()

    a.limpiar_frames()
    a.extraer_frames()        # ◀ carga TODOS los frames a memoria una sola vez

    a.analizar_frames_ipb()
    a.analizar_nitidez()
    a.analizar_fft()
    a.analizar_flujo_optico()
    a.analizar_fotometria()
    a.analizar_aberracion_cromatica()
    a.analizar_compresion()
    a.analizar_ruido()
    a.analizar_correlacion_rgb()

    a.predecir_con_modelo()
    a.generar_reporte_final()

    print("\n[✓] Análisis completado.")