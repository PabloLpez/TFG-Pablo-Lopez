"""
testear_holdout_real.py
------------------------
Evalúa el holdout usando EXACTAMENTE el mismo pipeline que la app web:
  - Extrae features con AnalizadorVideo (main42.py)
  - Enruta al modelo especializado según origen (enrutador_modelos.py)
  - Reporta métricas por origen y globales

Esta es la evaluación más fiable del sistema real, ya que usa
el mismo código de extracción de features que la aplicación web.

Uso:
    python testear_holdout_real.py --dir_holdout holdout --umbral 0.5

Estructura esperada del holdout:
    holdout/
    ├── real/
    │   ├── instagram/
    │   ├── twitter/
    │   └── movil_directo/
    └── fake/
        ├── instagram/
        ├── twitter/
        └── movil_directo/
"""

import os, glob, argparse, types, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve)

from main42 import AnalizadorVideo, FACTORES_ORIGEN
from enrutador_modelos import seleccionar_modelo

RUTA_MODELO_FALLBACK = 'modelo_v2.pkl'

# Necesario para deserializar los pkl entrenados con entrenar_ia5.py
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


# ─────────────────────────────────────────────────────────────
#  DESCUBRIR VÍDEOS DEL HOLDOUT
# ─────────────────────────────────────────────────────────────

def descubrir_videos(dir_holdout):
    """Devuelve lista de (ruta, label, origen)."""
    dir_holdout = Path(dir_holdout)
    videos = []
    for clase, label in [('real', 0), ('fake', 1)]:
        clase_dir = dir_holdout / clase
        if not clase_dir.exists():
            print(f"[!] No existe {clase_dir}")
            continue
        for origen_dir in sorted(clase_dir.iterdir()):
            if not origen_dir.is_dir():
                continue
            origen = origen_dir.name
            for ext in ('*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm'):
                for vid in sorted(origen_dir.glob(ext)):
                    videos.append((str(vid), label, origen))
    return videos


# ─────────────────────────────────────────────────────────────
#  ANALIZAR UN VÍDEO CON EL MISMO PIPELINE QUE LA APP
# ─────────────────────────────────────────────────────────────

def analizar_video(ruta_video, origen_manual='auto', fps=5, segundos=None):
    """
    Extrae features y obtiene predicción usando el pipeline de main42.py.
    Devuelve dict con prob_ia, tipo_modelo, origen_usado, informe completo.
    """
    dir_frames = f'_tmp_frames_{Path(ruta_video).stem}'
    os.makedirs(dir_frames, exist_ok=True)

    try:
        a = AnalizadorVideo.__new__(AnalizadorVideo)
        a.ruta              = ruta_video
        a.informe           = {}
        a.frames            = []
        a.frame_color       = None
        a.total_frames      = 0
        a.segundos_analizar = segundos
        a._dir              = dir_frames

        def _ruta_frame(self, i):
            return os.path.join(self._dir, f"out{i}.png")
        def _limpiar(self):
            for f in glob.glob(os.path.join(self._dir, 'out*.png')):
                os.remove(f)
        a._ruta_frame    = types.MethodType(_ruta_frame, a)
        a.limpiar_frames = types.MethodType(_limpiar, a)

        # Detectar origen
        a.detectar_origen()
        if origen_manual and origen_manual != 'auto':
            a.informe['origen_detectado'] = origen_manual

        # Seleccionar modelo
        origen_final = a.informe.get('origen_detectado', 'desconocido')
        ruta_modelo, tipo_modelo = seleccionar_modelo(origen_final,
                                                       fallback=RUTA_MODELO_FALLBACK)

        # Cargar modelo
        a.cargar_modelo(ruta_modelo)

        # Metadatos
        a.analizar_metadatos()
        a.analizar_metadatos_forenses()

        # Extraer frames
        args_t = ['-t', str(segundos)] if segundos else []
        subprocess.run(
            ['ffmpeg', '-y'] + args_t + [
                '-i', ruta_video,
                '-vf', f'fps={fps},scale=640:-2',
                os.path.join(dir_frames, 'out%d.png')
            ],
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
        a.frame_color  = cv2.imread(os.path.join(dir_frames, 'out1.png')) \
                         if a.total_frames else None

        if a.total_frames == 0:
            return None

        # Pipeline de análisis (igual que app.py)
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

        prob = a.informe.get('ml_prob_ia', 0.0)
        return {
            'prob_ia':      prob,
            'tipo_modelo':  tipo_modelo,
            'origen_usado': origen_final,
            'ruta_modelo':  ruta_modelo,
            'frames':       a.total_frames,
        }

    except Exception as e:
        print(f"    [!] Error en {Path(ruta_video).name}: {e}")
        return None
    finally:
        import shutil
        shutil.rmtree(dir_frames, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
#  MÉTRICAS Y REPORTE
# ─────────────────────────────────────────────────────────────

def metricas_origen(df, umbral):
    print(f"\n  DESGLOSE POR ORIGEN:")
    print(f"  {'origen':18s} {'n':>4s}  {'acc':>5s}  {'prec':>5s}  "
          f"{'rec':>5s}  {'AUC':>5s}  {'modelo':>12s}")
    print(f"  {'-'*18} {'-'*4}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*12}")
    for o in sorted(df['origen'].unique()):
        sub = df[df['origen'] == o]
        y   = sub['label'].values
        p   = sub['prob_ia'].values
        pred = (p >= umbral).astype(int)
        acc  = accuracy_score(y, pred)
        try:
            prec = precision_score(y, pred, zero_division=0)
            rec  = recall_score(y, pred, zero_division=0)
            auc  = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
        except Exception:
            prec = rec = auc = float('nan')
        mod = sub['tipo_modelo'].mode()[0] if 'tipo_modelo' in sub.columns else '?'
        print(f"  {o:18s} {len(sub):>4d}  {acc:.3f}  {prec:.3f}  "
              f"{rec:.3f}  {auc:.3f}  {mod:>12s}")


def generar_graficas(df, umbral, salida):
    """Genera 3 figuras: métricas globales, matriz confusión, desglose origen."""
    y    = df['label'].values
    prob = df['prob_ia'].values
    pred = (prob >= umbral).astype(int)

    if len(np.unique(y)) < 2:
        return

    base = Path(salida).with_suffix('')
    base.mkdir(exist_ok=True)

    # ─── FIGURA 1: métricas globales (ROC + PR + distribución) ────
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    axs[0].plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    axs[0].plot([0,1],[0,1],'--', color='gray', alpha=0.7)
    axs[0].set_xlabel('FPR'); axs[0].set_ylabel('TPR')
    axs[0].set_title('Curva ROC')
    axs[0].legend(); axs[0].grid(alpha=0.3)

    p_vals, r_vals, _ = precision_recall_curve(y, prob)
    axs[1].plot(r_vals, p_vals, lw=2)
    axs[1].set_xlabel('Recall'); axs[1].set_ylabel('Precision')
    axs[1].set_title('Curva Precision-Recall')
    axs[1].grid(alpha=0.3)

    axs[2].hist(prob[y==0], bins=15, alpha=0.6, label='Real (y=0)',
                color='steelblue', edgecolor='black')
    axs[2].hist(prob[y==1], bins=15, alpha=0.6, label='IA (y=1)',
                color='crimson', edgecolor='black')
    axs[2].axvline(umbral, color='black', ls='--', label=f'umbral={umbral}')
    axs[2].set_xlabel('P(IA)'); axs[2].set_ylabel('frecuencia')
    axs[2].set_title('Distribución P(IA) por clase')
    axs[2].legend()

    plt.tight_layout()
    out = base / 'metricas_globales.png'
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  [✓] {out}")

    # ─── FIGURA 2: matriz de confusión global ───────────────────
    def _plot_cm(ax, y_true, y_pred, titulo):
        """Dibuja una matriz de confusión en el eje dado."""
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real', 'IA/Falso'])
        ax.set_yticklabels(['Real', 'IA/Falso'])
        ax.set_xlabel('Predicción'); ax.set_ylabel('Etiqueta real')
        ax.set_title(titulo)
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=18, fontweight='bold')
        return im

    fig, ax = plt.subplots(figsize=(6, 5))
    im = _plot_cm(ax, y, pred, f'Matriz de confusión global (umbral={umbral})')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    out = base / 'matriz_confusion.png'
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  [✓] {out}")

    # ─── FIGURA 2b: matrices de confusión por plataforma ─────────
    if 'origen' in df.columns:
        origenes_cm = sorted(df['origen'].unique())
        n_cols = min(len(origenes_cm), 3)
        n_rows = (len(origenes_cm) + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 5 * n_rows),
                                 squeeze=False)
        for idx, o in enumerate(origenes_cm):
            sub  = df[df['origen'] == o]
            yo   = sub['label'].values
            preo = (sub['prob_ia'].values >= umbral).astype(int)
            ax_  = axs[idx // n_cols][idx % n_cols]
            try:
                im_ = _plot_cm(ax_, yo, preo,
                               f'{o}\n(n={len(sub)}, umbral={umbral})')
                fig.colorbar(im_, ax=ax_)
            except Exception:
                ax_.set_title(f'{o} — sin datos suficientes')
        # Ocultar ejes sobrantes
        for idx in range(len(origenes_cm), n_rows * n_cols):
            axs[idx // n_cols][idx % n_cols].set_visible(False)
        plt.suptitle('Matrices de confusión por plataforma', fontsize=14, y=1.02)
        plt.tight_layout()
        out = base / 'matrices_confusion_por_origen.png'
        plt.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [✓] {out}")

    # ─── FIGURA 3: desglose por origen ───────────────────────────
    if 'origen' in df.columns:
        origenes = sorted(df['origen'].unique())
        aucs, accs, ns = [], [], []
        for o in origenes:
            sub = df[df['origen'] == o]
            yo = sub['label'].values
            po = sub['prob_ia'].values
            preo = (po >= umbral).astype(int)
            ns.append(len(sub))
            accs.append((yo == preo).mean())
            aucs.append(roc_auc_score(yo, po) if len(np.unique(yo)) > 1 else float('nan'))

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(origenes))
        w = 0.35

        bars_auc = axs[0].bar(x - w/2, aucs, w, label='AUC',      color='#3498db', edgecolor='black')
        bars_acc = axs[0].bar(x + w/2, accs, w, label='Accuracy', color='#2ecc71', edgecolor='black')
        axs[0].set_xticks(x); axs[0].set_xticklabels(origenes)
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Métrica')
        axs[0].set_title('Rendimiento por plataforma (holdout)')
        axs[0].legend(); axs[0].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars_auc, aucs):
            if not np.isnan(val):
                axs[0].text(bar.get_x()+bar.get_width()/2, val+0.02,
                            f'{val:.2f}', ha='center', fontsize=9)
        for bar, val in zip(bars_acc, accs):
            axs[0].text(bar.get_x()+bar.get_width()/2, val+0.02,
                        f'{val:.2f}', ha='center', fontsize=9)
        axs[0].axhline(0.5, ls='--', color='gray', alpha=0.5)

        # Tamaños de muestra
        axs[1].bar(x, ns, color='#95a5a6', edgecolor='black')
        axs[1].set_xticks(x); axs[1].set_xticklabels(origenes)
        axs[1].set_ylabel('N vídeos')
        axs[1].set_title('Tamaño de muestra por plataforma')
        axs[1].grid(axis='y', alpha=0.3)
        for i, val in enumerate(ns):
            axs[1].text(i, val + max(ns)*0.02, str(val),
                        ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        out = base / 'desglose_origen.png'
        plt.savefig(str(out), dpi=150)
        plt.close()
        print(f"  [✓] {out}")

    # ─── FIGURA 4: métricas individualizadas por origen ─────────
    if 'origen' in df.columns:
        origenes = sorted(df['origen'].unique())
        n_orig = len(origenes)
        if n_orig >= 1:
            fig, axs = plt.subplots(n_orig, 3, figsize=(18, 5 * n_orig),
                                     squeeze=False)
            for i, o in enumerate(origenes):
                sub = df[df['origen'] == o]
                yo  = sub['label'].values
                po  = sub['prob_ia'].values

                # ROC por origen
                if len(np.unique(yo)) > 1:
                    fpr_o, tpr_o, _ = roc_curve(yo, po)
                    auc_o = roc_auc_score(yo, po)
                    axs[i, 0].plot(fpr_o, tpr_o, lw=2,
                                    label=f'AUC = {auc_o:.3f}')
                else:
                    axs[i, 0].text(0.5, 0.5, 'Solo una clase\n(no calculable)',
                                    ha='center', va='center',
                                    transform=axs[i, 0].transAxes)
                axs[i, 0].plot([0, 1], [0, 1], '--', color='gray', alpha=0.7)
                axs[i, 0].set_xlabel('FPR'); axs[i, 0].set_ylabel('TPR')
                axs[i, 0].set_title(f'ROC — {o} (n={len(sub)})')
                axs[i, 0].legend(); axs[i, 0].grid(alpha=0.3)

                # PR por origen
                if len(np.unique(yo)) > 1:
                    p_o, r_o, _ = precision_recall_curve(yo, po)
                    axs[i, 1].plot(r_o, p_o, lw=2)
                axs[i, 1].set_xlabel('Recall'); axs[i, 1].set_ylabel('Precision')
                axs[i, 1].set_title(f'Precision-Recall — {o}')
                axs[i, 1].grid(alpha=0.3)
                axs[i, 1].set_xlim(0, 1); axs[i, 1].set_ylim(0, 1.05)

                # Distribución por origen
                if (yo == 0).any():
                    axs[i, 2].hist(po[yo == 0], bins=10, alpha=0.6,
                                    label=f'Real ({(yo==0).sum()})',
                                    color='steelblue', edgecolor='black')
                if (yo == 1).any():
                    axs[i, 2].hist(po[yo == 1], bins=10, alpha=0.6,
                                    label=f'IA ({(yo==1).sum()})',
                                    color='crimson', edgecolor='black')
                axs[i, 2].axvline(umbral, color='black', ls='--',
                                   label=f'umbral={umbral}')
                axs[i, 2].set_xlabel('P(IA)'); axs[i, 2].set_ylabel('frecuencia')
                axs[i, 2].set_title(f'Distribución P(IA) — {o}')
                axs[i, 2].legend()

            plt.tight_layout()
            out = base / 'metricas_por_origen.png'
            plt.savefig(str(out), dpi=150)
            plt.close()
            print(f"  [✓] {out}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir_holdout', default='holdout',
                    help="Carpeta raíz del holdout (contiene real/ y fake/)")
    ap.add_argument('--umbral',      type=float, default=0.5)
    ap.add_argument('--fps',         type=int,   default=5)
    ap.add_argument('--segundos',    type=int,   default=None)
    ap.add_argument('--salida',      default='pred_holdout_real.csv')
    args = ap.parse_args()

    videos = descubrir_videos(args.dir_holdout)
    if not videos:
        print("[!] No se encontraron vídeos en el holdout.")
        return

    print(f"\n[+] {len(videos)} vídeos encontrados en {args.dir_holdout}")
    print(f"    Reales: {sum(1 for _,l,_ in videos if l==0)} | "
          f"Fakes: {sum(1 for _,l,_ in videos if l==1)}")

    resultados = []
    errores    = []

    for i, (ruta, label, origen) in enumerate(videos, 1):
        nombre = Path(ruta).name
        print(f"  [{i:2d}/{len(videos)}] {nombre[:60]:60s}", end=' ', flush=True)

        # Forzar el origen según la carpeta donde está el vídeo
        # (replica lo que hace el usuario seleccionando plataforma en la app)
        res = analizar_video(ruta, origen_manual=origen,
                              fps=args.fps, segundos=args.segundos)

        if res is None:
            print("ERROR")
            errores.append(ruta)
            continue

        pred = int(res['prob_ia'] >= args.umbral)
        ok   = '✓' if pred == label else '✗'
        print(f"P(IA)={res['prob_ia']:.3f}  pred={pred}  label={label}  "
              f"{ok}  [{res['tipo_modelo']}]")

        resultados.append({
            'video':       ruta,
            'origen':      origen,
            'label':       label,
            'prob_ia':     res['prob_ia'],
            'pred':        pred,
            'confianza':   abs(res['prob_ia'] - 0.5) * 2,
            'error':       int(pred != label),
            'tipo_modelo': res['tipo_modelo'],
            'origen_usado':res['origen_usado'],
            'frames':      res['frames'],
        })

    if not resultados:
        print("[!] Sin resultados.")
        return

    df = pd.DataFrame(resultados)
    df.to_csv(args.salida, index=False)
    print(f"\n[✓] Predicciones guardadas: {args.salida}")

    # ── Métricas globales
    y    = df['label'].values
    prob = df['prob_ia'].values
    pred = df['pred'].values

    print("\n" + "═"*64)
    print(" MÉTRICAS GLOBALES (pipeline real) ".center(64, "═"))
    print("═"*64)
    print(classification_report(y, pred, target_names=['Real','IA/Falso']))
    print(f"  Confusión:\n{confusion_matrix(y, pred)}")
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, prob)
        print(f"  ROC-AUC: {auc:.4f}")

    metricas_origen(df, args.umbral)

    # Errores de alta confianza
    errores_df = df[df['error']==1].sort_values('confianza', ascending=False)
    if len(errores_df) > 0:
        print(f"\n  ERRORES DE ALTA CONFIANZA ({len(errores_df)} errores totales):")
        cols = ['video','origen','label','pred','prob_ia','confianza','tipo_modelo']
        print(errores_df[cols].head(10).to_string(index=False))

    generar_graficas(df, args.umbral, args.salida)

    if errores:
        print(f"\n[!] {len(errores)} vídeos fallaron: {errores}")


if __name__ == '__main__':
    main()