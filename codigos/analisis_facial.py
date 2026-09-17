"""
analisis_facial.py
-------------------
Módulo de análisis facial para detección de IA en primeros planos.
Usa MediaPipe Face Landmarker (nueva Tasks API, compatible con mediapipe>=0.10).
Descarga automáticamente el modelo la primera vez (~3MB).

Features generadas (9):
  face_detectada           : 1.0 si hay cara en >50% de frames, 0.0 si no
  face_parpadeo_freq       : parpadeos por minuto (humano normal: 15-20)
  face_parpadeo_simetria   : descoordinación entre ojos (0=perfecta)
  face_boca_apertura_std   : variación temporal de apertura de boca
  face_landmark_temblor    : temblor medio de landmarks entre frames
  face_eye_aspect_std      : variación del eye aspect ratio
  face_simetria_facial     : asimetría facial media (0=simétrico)
  face_borde_estabilidad   : inestabilidad del contorno facial
  face_iluminacion_consist : consistencia de iluminación cara-fondo

Uso standalone:
    python analisis_facial.py --video clip.mp4

Uso programático:
    from analisis_facial import calcular_features_faciales
    feats = calcular_features_faciales(frames_color)
"""

import os, sys, argparse, subprocess, tempfile, shutil, glob, urllib.request
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
#  MODELO MEDIAPIPE (se descarga automáticamente si no existe)
# ─────────────────────────────────────────────────────────────

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _descargar_modelo():
    if os.path.exists(MODEL_PATH):
        return True
    try:
        print(f"[+] Descargando modelo MediaPipe (~3 MB)…", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[+] Modelo guardado en {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[!] No se pudo descargar el modelo: {e}", file=sys.stderr)
        return False


def _crear_detector():
    """Crea el detector de landmarks faciales con la nueva Tasks API."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("[!] MediaPipe no instalado. Ejecuta: pip install mediapipe",
              file=sys.stderr)
        return None

    if not _descargar_modelo():
        return None

    try:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[!] Error creando detector MediaPipe: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────
#  DEFINICIONES
# ─────────────────────────────────────────────────────────────

NOMBRES_FEATURES_FACIALES = [
    'face_detectada',
    'face_parpadeo_freq',
    'face_parpadeo_simetria',
    'face_boca_apertura_std',
    'face_landmark_temblor',
    'face_eye_aspect_std',
    'face_simetria_facial',
    'face_borde_estabilidad',
    'face_iluminacion_consist',
]

FEATURES_VACIAS = {k: 0.0 for k in NOMBRES_FEATURES_FACIALES}

# Índices MediaPipe Face Mesh 478 puntos
OJO_IZQ_EAR  = [33,  160, 158, 133, 153, 144]
OJO_DER_EAR  = [362, 385, 387, 263, 373, 380]
BOCA_SUP, BOCA_INF = 13, 14
BOCA_IZQ, BOCA_DER = 78, 308
CONTORNO = [10,338,297,332,284,251,389,356,454,323,361,
            288,397,365,379,378,400,377,152,148,176,149,
            150,136,172, 58,132, 93,234,127,162, 21, 54,103,67,109]
PARES_SIM = [(33,263),(133,362),(61,291),(78,308),(234,454)]


# ─────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────

def _ear(lms, idx):
    p = [lms[i] for i in idx]
    a = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    b = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    c = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (a + b) / (2.0 * c) if c > 1e-6 else 0.0


def _contar_parpadeos(serie, umbral=0.21):
    count, dentro = 0, False
    for v in serie:
        if v < umbral and not dentro:
            count += 1; dentro = True
        elif v >= umbral:
            dentro = False
    return count


# ─────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def calcular_features_faciales(frames_color, fps_video=5):
    """
    Calcula features faciales sobre una lista de frames BGR.
    Devuelve dict con las 9 features.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return dict(FEATURES_VACIAS)

    if not frames_color or len(frames_color) < 3:
        return dict(FEATURES_VACIAS)

    detector = _crear_detector()
    if detector is None:
        return dict(FEATURES_VACIAS)

    ear_i_s, ear_d_s = [], []
    boca_s, sim_s    = [], []
    temblores         = []
    contornos         = []
    ilum_s            = []
    lms_ant           = None
    n_cara            = 0

    try:
        for frame in frames_color:
            if frame is None or frame.size == 0:
                continue
            h, w = frame.shape[:2]

            # Convertir a RGB y crear imagen MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)

            if not res.face_landmarks:
                lms_ant = None
                continue

            n_cara += 1
            raw = res.face_landmarks[0]
            lms = [(lm.x * w, lm.y * h) for lm in raw]

            # ── EAR
            ear_i_s.append(_ear(lms, OJO_IZQ_EAR))
            ear_d_s.append(_ear(lms, OJO_DER_EAR))

            # ── Apertura boca
            vert = np.linalg.norm(np.array(lms[BOCA_SUP]) - np.array(lms[BOCA_INF]))
            anch = np.linalg.norm(np.array(lms[BOCA_IZQ]) - np.array(lms[BOCA_DER]))
            if anch > 1e-6:
                boca_s.append(vert / anch)

            # ── Simetría
            cx = lms[1][0]
            asim = []
            for i1, i2 in PARES_SIM:
                d1 = abs(lms[i1][0] - cx)
                d2 = abs(lms[i2][0] - cx)
                if max(d1, d2) > 1e-6:
                    asim.append(abs(d1 - d2) / max(d1, d2))
            if asim:
                sim_s.append(float(np.mean(asim)))

            # ── Temblor de landmarks
            lms_arr = np.array(lms)
            if lms_ant is not None:
                ancho_cara = np.linalg.norm(np.array(lms[234]) - np.array(lms[454]))
                if ancho_cara > 1e-6:
                    mov = np.mean(np.linalg.norm(lms_arr - lms_ant, axis=1))
                    temblores.append(mov / ancho_cara)
            lms_ant = lms_arr

            # ── Contorno
            contornos.append(np.array([lms[i] for i in CONTORNO]))

            # ── Iluminación cara vs fondo
            xs = [int(p[0]) for p in lms]
            ys = [int(p[1]) for p in lms]
            x0, x1 = max(0, min(xs)), min(w, max(xs))
            y0, y1 = max(0, min(ys)), min(h, max(ys))
            if x1 > x0 + 10 and y1 > y0 + 10:
                cara_reg = frame[y0:y1, x0:x1]
                mask     = np.ones((h, w), dtype=bool)
                mask[y0:y1, x0:x1] = False
                if mask.sum() > 100:
                    bc = cv2.cvtColor(cara_reg, cv2.COLOR_BGR2GRAY).mean()
                    bf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[mask].mean()
                    if bf > 1e-6:
                        ilum_s.append(abs(bc - bf) / bf)
    finally:
        detector.close()

    n = len(frames_color)
    if n_cara / max(n, 1) < 0.5:
        return {**FEATURES_VACIAS, 'face_detectada': 0.0}

    dur = n / max(fps_video, 1)
    pb  = _contar_parpadeos(ear_i_s)
    pd  = _contar_parpadeos(ear_d_s)
    ppm = (pb + pd) / 2 * (60.0 / max(dur, 1e-6))

    todos_ear = ear_i_s + ear_d_s
    if todos_ear and ear_i_s:
        ei = np.array(ear_i_s); ed = np.array(ear_d_s)
        denom = (ei + ed).mean() + 1e-6
        sim_parp = float(np.mean(np.abs(ei - ed)) / denom)
    else:
        sim_parp = 0.0

    if len(contornos) > 1:
        movs = [np.mean(np.linalg.norm(contornos[k] - contornos[k-1], axis=1))
                for k in range(1, len(contornos))]
        borde_est = float(np.mean(movs))
    else:
        borde_est = 0.0

    return {
        'face_detectada':           1.0,
        'face_parpadeo_freq':       float(ppm),
        'face_parpadeo_simetria':   float(sim_parp),
        'face_boca_apertura_std':   float(np.std(boca_s)) if boca_s else 0.0,
        'face_landmark_temblor':    float(np.mean(temblores)) if temblores else 0.0,
        'face_eye_aspect_std':      float(np.std(todos_ear)) if todos_ear else 0.0,
        'face_simetria_facial':     float(np.mean(sim_s)) if sim_s else 0.0,
        'face_borde_estabilidad':   borde_est,
        'face_iluminacion_consist': float(np.std(ilum_s)) if ilum_s else 0.0,
    }


# ─────────────────────────────────────────────────────────────
#  EXTRACCIÓN DESDE ARCHIVO DE VÍDEO
# ─────────────────────────────────────────────────────────────

def extraer_features_de_video(ruta_video, fps=5, segundos=None):
    """Extrae frames con ffmpeg y calcula las features faciales."""
    dir_tmp = tempfile.mkdtemp(prefix='analisis_facial_')
    try:
        args_t = ['-t', str(segundos)] if segundos else []
        subprocess.run(
            ['ffmpeg', '-y'] + args_t + [
                '-i', ruta_video,
                '-vf', f'fps={fps},scale=640:-2',
                os.path.join(dir_tmp, 'out%d.png')
            ],
            capture_output=True
        )
        frames, i = [], 1
        while True:
            p = os.path.join(dir_tmp, f'out{i}.png')
            if not os.path.exists(p):
                break
            img = cv2.imread(p)
            if img is None:
                break
            frames.append(img)
            i += 1
        if not frames:
            print("[!] No se pudieron extraer frames del vídeo", file=sys.stderr)
            return dict(FEATURES_VACIAS)
        return calcular_features_faciales(frames, fps_video=fps)
    finally:
        shutil.rmtree(dir_tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
#  MAIN (uso standalone)
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',    required=True)
    ap.add_argument('--fps',      type=int, default=5)
    ap.add_argument('--segundos', type=int, default=None)
    args = ap.parse_args()

    print(f"\n[+] Analizando: {args.video}")
    feats = extraer_features_de_video(args.video, fps=args.fps, segundos=args.segundos)

    print("\nFeatures faciales:")
    print("─" * 50)
    for k, v in feats.items():
        flag = ''
        if k == 'face_detectada':
            flag = ' ← cara detectada' if v >= 1 else ' ← SIN CARA (resto = 0)'
        if k == 'face_parpadeo_freq' and feats.get('face_detectada', 0) >= 1:
            if   v < 5:   flag = ' ← muy bajo, sospechoso'
            elif v > 40:  flag = ' ← muy alto'
            else:         flag = ' ← rango humano normal (15-20/min)'
        print(f"  {k:32s} = {v:.4f}{flag}")


if __name__ == '__main__':
    main()
