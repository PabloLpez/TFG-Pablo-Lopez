"""
extraer_datos.py
-----------------
Extrae features de datasets/real/ y datasets/fake/ a un CSV.

Optimización clave: cada vídeo se lee del disco UNA vez y todos los
análisis comparten la misma lista de frames en memoria (4× menos IO).
"""

import os, json, argparse, subprocess, glob, csv, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from scipy.stats import shapiro
from scipy import stats
import shutil

RUTA_EXIFTOOL = os.environ.get('EXIFTOOL_PATH') or shutil.which('exiftool') or 'exiftool'
FPS, DIR_TMP  = 5, "frames_tmp"

ORIGENES = ['movil_directo', 'instagram', 'twitter', 'whatsapp', 'tiktok', 'desconocido']

NOMBRES_FEATURES = [
    "nitidez_media", "nitidez_std", "nitidez_outlier_pct",
    "fft_energia_media", "fft_ratio_alta_baja", "fft_ratio_std",
    "flujo_medio", "flujo_std", "flujo_salto_maximo", "flujo_num_glitches",
    "luz_std", "brillo_std", "brillo_tendencia",
    "ca_desfase_medio", "ca_desfase_std",
    "dct_ratio",
    "ruido_shapiro_p", "ruido_std", "ruido_asimetria", "ruido_curtosis",
    "corr_rg", "corr_rb", "corr_gb",
    "ipb_pct_i", "ipb_pct_p", "ipb_pct_b",
    "ipb_gop_medio", "ipb_gop_std", "ipb_ratio_tamano_ip", "ipb_num_alertas",
    "meta_num_alertas",
    "origen_movil_directo", "origen_instagram", "origen_twitter",
    "origen_whatsapp", "origen_tiktok", "origen_desconocido",
]


# ═════════════════════════════════════════════════════════════════
#  ORIGEN (yt-dlp)
# ═════════════════════════════════════════════════════════════════

def leer_origen_ydlp(ruta_video):
    json_p = os.path.splitext(ruta_video)[0] + '.info.json'
    if not os.path.exists(json_p):
        return 'movil_directo'
    try:
        extractor = json.load(open(json_p, encoding='utf-8')).get('extractor', '').lower()
    except Exception:
        return 'desconocido'
    for clave in ('instagram', 'twitter', 'whatsapp', 'tiktok'):
        if clave in extractor:
            return clave
    return 'twitter' if 'x.com' in extractor else 'desconocido'


def codificar_origen(origen):
    return {f'origen_{o}': int(origen == o) for o in ORIGENES}


# ═════════════════════════════════════════════════════════════════
#  FRAMES (lectura única)
# ═════════════════════════════════════════════════════════════════

def extraer_frames(ruta_video, dir_tmp, fps):
    os.makedirs(dir_tmp, exist_ok=True)
    subprocess.run(
        ['ffmpeg', '-y', '-i', ruta_video, '-vf', f'fps={fps}',
         os.path.join(dir_tmp, 'out%d.png')],
        capture_output=True
    )
    return len(glob.glob(os.path.join(dir_tmp, 'out*.png')))


def cargar_frames(dir_tmp):
    """Lee TODOS los frames a memoria una sola vez."""
    frames, i = [], 1
    while True:
        p = os.path.join(dir_tmp, f'out{i}.png')
        if not os.path.exists(p):      # evita el warning de OpenCV y el delay
            break
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            break
        frames.append(img)
        i += 1
    color = cv2.imread(os.path.join(dir_tmp, 'out1.png')) if frames else None
    return frames, color


def limpiar_frames(dir_tmp):
    for f in glob.glob(os.path.join(dir_tmp, 'out*.png')):
        os.remove(f)


# ═════════════════════════════════════════════════════════════════
#  CÁLCULOS (reciben frames ya en memoria)
# ═════════════════════════════════════════════════════════════════

def calc_nitidez(frames):
    if not frames:
        return 0.0, 0.0, 0.0
    vals = np.array([cv2.Laplacian(f, cv2.CV_64F).var() for f in frames])
    mu, sg = float(vals.mean()), float(vals.std())
    return mu, sg, float(np.mean(np.abs(vals - mu) > 2 * sg))


def calc_fft(frames):
    if not frames:
        return 0.0, 0.0, 0.0
    h, w = frames[0].shape
    mascara = np.zeros((h, w))
    cv2.circle(mascara, (w // 2, h // 2), min(h, w) // 8, 1, -1)
    inv = 1 - mascara
    energias, ratios = [], []
    for img in frames:
        mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1)
        energias.append(mag.mean())
        ratios.append((mag * inv).sum() / ((mag * mascara).sum() + 1e-6))
    return float(np.mean(energias)), float(np.mean(ratios)), float(np.std(ratios))


def calc_flujo_optico(frames):
    if len(frames) < 2:
        return 0.0, 0.0, 0.0, 0
    h, w = frames[0].shape
    movs, saltos, glitches = [], [], 0
    prev = frames[0]
    for curr in frames[1:]:
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
    return (float(np.mean(movs)), float(np.std(movs)),
            float(max(saltos)) if saltos else 0.0, glitches)


def calc_fotometria(frames):
    if not frames:
        return 0.0, 0.0, 0.0
    direcciones, brillos = [], []
    for img in frames:
        bl = cv2.GaussianBlur(img, (15, 15), 0)
        sx = cv2.Sobel(bl, cv2.CV_64F, 1, 0, ksize=5)
        sy = cv2.Sobel(bl, cv2.CV_64F, 0, 1, ksize=5)
        direcciones.append(float(np.arctan2(sy, sx).mean() * 180 / np.pi))
        brillos.append(float(img.mean()))
    tend = float(np.polyfit(range(len(brillos)), brillos, 1)[0])
    return float(np.std(direcciones)), float(np.std(brillos)), tend


def _parche(canal, y, x, t=100):
    p = canal[y - t // 2:y + t // 2, x - t // 2:x + t // 2]
    return p if p.shape == (t, t) else None


def calc_aberracion_cromatica(color):
    if color is None:
        return 0.0, 0.0
    _, g, r = cv2.split(color.astype(np.float32))
    h, w = color.shape[:2]
    t = 100
    desps = []
    for y, x in [(h//2, w//2), (t, t), (t, w-t), (h-t, t), (h-t, w-t)]:
        pg, pr = _parche(g, y, x), _parche(r, y, x)
        if pg is None or pr is None or pg.sum() == 0:
            continue
        shift, _ = cv2.phaseCorrelate(pg, pr)
        desps.append(float(np.hypot(*shift)))
    if not desps:
        return 0.0, 0.0
    return float(np.mean(desps)), float(np.std(desps))


def calc_dct(frames):
    if not frames:
        return 0.0
    img = frames[0]
    gx = np.abs(np.diff(img.astype(float), axis=1))
    gy = np.abs(np.diff(img.astype(float), axis=0))
    mgx, mgy = gx.mean(), gy.mean()

    def bloques(bs):
        h, w = img.shape
        c  = sum(1 for i in range(bs, h, bs) if i < gy.shape[0] and gy[i].mean() > mgy * 1.2)
        c += sum(1 for j in range(bs, w, bs) if j < gx.shape[1] and gx[:, j].mean() > mgx * 1.2)
        return c

    return float(bloques(8) / (bloques(16) + 1))


def calc_ruido(frames):
    if not frames:
        return 0.0, 0.0, 0.0, 0.0
    img   = frames[0]
    ruido = img.astype(float) - cv2.medianBlur(img, 5).astype(float)
    flat  = ruido.flatten()
    muestra = np.random.choice(flat, 5000, replace=False) if len(flat) > 5000 else flat
    _, p = shapiro(muestra)
    return float(p), float(ruido.std()), float(stats.skew(flat)), float(stats.kurtosis(flat))


def calc_correlacion_rgb(color):
    if color is None:
        return 0.0, 0.0, 0.0
    b, g, r = cv2.split(color)
    rf, gf, bf = r.flatten(), g.flatten(), b.flatten()
    return (float(np.corrcoef(rf, gf)[0, 1]),
            float(np.corrcoef(rf, bf)[0, 1]),
            float(np.corrcoef(gf, bf)[0, 1]))


def calc_ipb(ruta_video):
    cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0', '-show_frames',
           '-show_entries', 'frame=pict_type,pkt_size', '-of', 'json', ruta_video]
    res = subprocess.run(cmd, capture_output=True, text=True)
    frames = json.loads(res.stdout).get('frames', []) if res.stdout.strip() else []
    if not frames:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

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

    alertas = sum([conteos['B'] == 0, pct['I'] > 20,
                   gs < 1.0 and gm > 0, rip < 1.5])
    return pct['I'], pct['P'], pct['B'], gm, gs, rip, alertas


def calc_metadatos(ruta_video):
    res = subprocess.run([RUTA_EXIFTOOL, '-j', ruta_video], capture_output=True, text=True)
    if not res.stdout.strip():
        return 3
    meta = json.loads(res.stdout)[0]
    a = 0
    fab = meta.get('Make', '').lower()
    if fab in {'unknown', 'adobe', 'synthesia', 'runway', 'generic', ''} or 'ai' in fab:
        a += 1
    if sum(1 for c in ('Make', 'Model', 'CreateDate', 'Software') if not meta.get(c)) >= 3:
        a += 1
    fc, fm = meta.get('CreateDate'), meta.get('ModifyDate')
    if fc and fm and fc > fm:
        a += 1
    return a


# ═════════════════════════════════════════════════════════════════
#  PROCESADOR POR VÍDEO
# ═════════════════════════════════════════════════════════════════

def procesar_video(args):
    ruta_video, etiqueta, fps = args
    dir_tmp = f"{DIR_TMP}_{os.getpid()}"

    try:
        if extraer_frames(ruta_video, dir_tmp, fps) == 0:
            return None

        frames, color = cargar_frames(dir_tmp)   # ◀ una sola lectura
        if not frames:
            return None

        nit_med, nit_std, nit_out       = calc_nitidez(frames)
        fft_e, fft_r, fft_rs            = calc_fft(frames)
        fl_med, fl_std, fl_salto, fl_gl = calc_flujo_optico(frames)
        luz_std, br_std, br_tend        = calc_fotometria(frames)
        ca_med, ca_std                  = calc_aberracion_cromatica(color)
        dct_r                           = calc_dct(frames)
        r_p, r_std, r_asim, r_curt      = calc_ruido(frames)
        rg, rb, gb                      = calc_correlacion_rgb(color)
        pi, pp, pb, gm, gs, ir, ial     = calc_ipb(ruta_video)
        mal                             = calc_metadatos(ruta_video)
        origen                          = leer_origen_ydlp(ruta_video)

        return {
            'video': ruta_video, 'etiqueta': etiqueta, 'origen': origen,
            'nitidez_media': nit_med, 'nitidez_std': nit_std, 'nitidez_outlier_pct': nit_out,
            'fft_energia_media': fft_e, 'fft_ratio_alta_baja': fft_r, 'fft_ratio_std': fft_rs,
            'flujo_medio': fl_med, 'flujo_std': fl_std,
            'flujo_salto_maximo': fl_salto, 'flujo_num_glitches': fl_gl,
            'luz_std': luz_std, 'brillo_std': br_std, 'brillo_tendencia': br_tend,
            'ca_desfase_medio': ca_med, 'ca_desfase_std': ca_std,
            'dct_ratio': dct_r,
            'ruido_shapiro_p': r_p, 'ruido_std': r_std,
            'ruido_asimetria': r_asim, 'ruido_curtosis': r_curt,
            'corr_rg': rg, 'corr_rb': rb, 'corr_gb': gb,
            'ipb_pct_i': pi, 'ipb_pct_p': pp, 'ipb_pct_b': pb,
            'ipb_gop_medio': gm, 'ipb_gop_std': gs,
            'ipb_ratio_tamano_ip': ir, 'ipb_num_alertas': ial,
            'meta_num_alertas': mal,
            **codificar_origen(origen),
        }

    except Exception as e:
        print(f"  [!] Error en {ruta_video}: {e}")
        traceback.print_exc()
        return None
    finally:
        limpiar_frames(dir_tmp)
        try:
            os.rmdir(dir_tmp)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════

def buscar_videos(carpeta):
    exts  = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    found = set()
    for ext in exts:
        found.update(glob.glob(os.path.join(carpeta, ext)))
        found.update(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fps',      type=int, default=FPS)
    ap.add_argument('--workers',  type=int, default=2)
    ap.add_argument('--dir_real', default='datasets/real')
    ap.add_argument('--dir_fake', default='datasets/fake')
    ap.add_argument('--salida',   default='features.csv')
    args = ap.parse_args()

    reales = buscar_videos(args.dir_real)
    falsos = buscar_videos(args.dir_fake)
    print(f"[+] Reales: {len(reales)} | Falsos: {len(falsos)} | FPS: {args.fps} | Workers: {args.workers}\n")

    dist = {}
    for v in reales:
        o = leer_origen_ydlp(v)
        dist[o] = dist.get(o, 0) + 1
    print("[+] Orígenes en reales:")
    for o, n in sorted(dist.items()):
        print(f"    {o:20s}: {n}")
    print()

    tareas = [(v, 0, args.fps) for v in reales] + [(v, 1, args.fps) for v in falsos]
    cols   = ['video', 'etiqueta', 'origen'] + NOMBRES_FEATURES
    filas  = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futuros = {ex.submit(procesar_video, t): t for t in tareas}
        for n, fut in enumerate(as_completed(futuros), 1):
            r     = fut.result()
            tarea = futuros[fut]
            clase = "REAL" if tarea[1] == 0 else "FAKE"
            org   = r.get('origen', '?') if r else '?'
            est   = "OK"   if r else "ERROR"
            if r:
                filas.append(r)
            print(f"  [{n}/{len(tareas)}] {clase} | {org:15s} | {est} | {Path(tarea[0]).name}")

    with open(args.salida, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(filas)

    print(f"\n[✓] CSV guardado: {args.salida}")
    print(f"[✓] {len(filas)}/{len(tareas)} vídeos procesados.")


if __name__ == '__main__':
    main()