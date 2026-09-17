"""
entrenar_especializados.py
---------------------------
Entrena modelos especializados por plataforma: un modelo por cada origen
del dataset. Comparamos los modelos especializados con el modelo unificado.

Workflow:
  1. Lee features_v2.csv
  2. Filtra por cada origen, genera features_<origen>.csv
  3. Entrena un modelo por origen (modelo_<origen>.pkl)
  4. Genera tabla comparativa: AUC unificado vs AUC especializado por plataforma

Uso:
  python entrenar_especializados.py --csv features_v2.csv --modelo_unificado modelo_v2.pkl
"""

import argparse, subprocess, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Necesario para deserializar los modelos entrenados con entrenar_ia5.py
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',               default='features_v2.csv')
    ap.add_argument('--modelo_unificado',  default='modelo_v2.pkl')
    ap.add_argument('--script_entreno',    default='entrenar_ia5.py')
    ap.add_argument('--salida',            default='especializados')
    ap.add_argument('--min_muestras',      type=int, default=40,
                    help="Mínimo de vídeos por origen para entrenar modelo especializado")
    args = ap.parse_args()

    Path(args.salida).mkdir(exist_ok=True)

    # 1. Leer dataset completo
    df = pd.read_csv(args.csv)
    if 'origen' not in df.columns:
        print("[!] El CSV no tiene columna 'origen'")
        return

    origenes = df['origen'].unique()
    print(f"[+] Orígenes encontrados: {list(origenes)}")
    print(f"[+] Distribución:")
    print(df['origen'].value_counts())

    # 2. Filtrar y entrenar por origen
    modelos_entrenados = []
    for origen in origenes:
        sub = df[df['origen'] == origen]
        n_real = (sub['etiqueta'] == 0).sum() if 'etiqueta' in sub.columns else (sub['label'] == 0).sum()
        n_fake = (sub['etiqueta'] == 1).sum() if 'etiqueta' in sub.columns else (sub['label'] == 1).sum()

        if len(sub) < args.min_muestras:
            print(f"\n[!] '{origen}' tiene {len(sub)} vídeos (<{args.min_muestras}), se omite")
            continue
        if min(n_real, n_fake) < 10:
            print(f"\n[!] '{origen}' tiene clase minoritaria <10 ({n_real}r/{n_fake}f), se omite")
            continue

        # Guardar CSV filtrado
        csv_filtrado = Path(args.salida) / f'features_{origen}.csv'
        sub.to_csv(csv_filtrado, index=False)

        # Modelo de salida
        modelo_salida = Path(args.salida) / f'modelo_{origen}.pkl'
        log_salida = Path(args.salida) / f'log_{origen}.txt'

        print(f"\n{'═'*64}")
        print(f" ENTRENANDO MODELO PARA '{origen}' ({len(sub)} vídeos) ".center(64, "═"))
        print(f"{'═'*64}")

        # Llamar a entrenar_ia5.py con CSV filtrado, saltando LOPO (solo 1 origen)
        # Usar path absoluto para evitar problemas de directorio de trabajo en Windows
        modelo_salida_abs = modelo_salida.resolve()
        csv_filtrado_abs  = csv_filtrado.resolve()

        cmd = ['python', args.script_entreno,
               '--csv',    str(csv_filtrado_abs),
               '--salida', str(modelo_salida_abs),
               '--saltar_lopo']

        with open(log_salida, 'w', encoding='utf-8') as flog:
            import os as _os
            env_utf8 = _os.environ.copy()
            env_utf8['PYTHONIOENCODING'] = 'utf-8'
            result = subprocess.run(cmd, capture_output=True, env=env_utf8)
            stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
            stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
            flog.write(stdout)
            if stderr:
                flog.write("\n\n--- STDERR ---\n" + stderr)
            print(stdout[-2000:])
            if result.returncode != 0:
                print(f"  [!] entrenar_ia5.py terminó con código {result.returncode}")
                if stderr:
                    print(f"  [!] Error:\n{stderr[:500]}")

        if modelo_salida_abs.exists():
            print(f"  [✓] Modelo guardado: {modelo_salida_abs}")
            modelos_entrenados.append({
                'origen': origen,
                'modelo': str(modelo_salida_abs),
                'csv':    str(csv_filtrado_abs),
                'n_real': int(n_real),
                'n_fake': int(n_fake),
            })
        else:
            print(f"  [!] MODELO NO ENCONTRADO en {modelo_salida_abs}")
            print(f"  [!] Revisa el log: {log_salida.resolve()}")

    # 3. Comparar contra el modelo unificado
    print(f"\n{'═'*64}")
    print(" COMPARATIVA UNIFICADO vs ESPECIALIZADOS ".center(64, "═"))
    print(f"{'═'*64}")

    if not modelos_entrenados:
        print("\n[!] No se encontró ningún modelo especializado.")
        print("    Revisa los archivos log_*.txt en la carpeta 'especializados/'")
        print("    para ver qué falló durante el entrenamiento.")
        return

    # Cargar unificado
    with open(args.modelo_unificado, 'rb') as f:
        d_unif = pickle.load(f)
    modelo_unif = d_unif['modelo']
    fnames = d_unif['feature_names']

    filas = []
    for info in modelos_entrenados:
        # Cargar modelo especializado
        with open(info['modelo'], 'rb') as f:
            d_esp = pickle.load(f)
        modelo_esp = d_esp['modelo']

        # Leer CSV del origen
        df_o = pd.read_csv(info['csv'])
        if 'etiqueta' in df_o.columns and 'label' not in df_o.columns:
            df_o = df_o.rename(columns={'etiqueta': 'label'})
        for c in fnames:
            if c not in df_o.columns:
                df_o[c] = np.nan
        X = df_o[fnames].values
        y = df_o['label'].values

        # Evaluar unificado y especializado sobre el mismo subset
        # (NOTA: ambos están entrenados con estos datos, así que es la métrica
        #  in-sample. La comparación rigurosa requeriría CV en cada uno.)
        prob_unif = modelo_unif.predict_proba(X)[:, 1]

        # Para el especializado, usamos sus features (puede haber excluido alguna)
        fnames_esp = d_esp['feature_names']
        for c in fnames_esp:
            if c not in df_o.columns:
                df_o[c] = np.nan
        X_esp = df_o[fnames_esp].values
        prob_esp = modelo_esp.predict_proba(X_esp)[:, 1]

        # CV del especializado (de su pkl)
        cv_esp = d_esp['cv_resultados'][d_esp['best_model_name']]

        filas.append({
            'origen':           info['origen'],
            'n_total':          len(df_o),
            'n_real':           info['n_real'],
            'n_fake':           info['n_fake'],
            'auc_unif_subset':  roc_auc_score(y, prob_unif),
            'auc_esp_subset':   roc_auc_score(y, prob_esp),
            'auc_esp_cv':       cv_esp['roc_auc'],
            'auc_esp_cv_std':   cv_esp.get('auc_std', 0),
            'modelo_esp':       d_esp['best_model_name'],
        })

    df_res = pd.DataFrame(filas)
    df_res['mejora'] = df_res['auc_esp_cv'] - df_res['auc_unif_subset']
    csv_out = Path(args.salida) / 'comparativa.csv'
    df_res.to_csv(csv_out, index=False)

    print("\n" + df_res.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f"\n[✓] Tabla guardada: {csv_out}")

    # 4. Gráfico comparativo
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_res))
    w = 0.35
    ax.bar(x - w/2, df_res['auc_unif_subset'], w, label='Unificado (in-sample)', color='#95a5a6')
    ax.bar(x + w/2, df_res['auc_esp_cv'], w,
           yerr=df_res['auc_esp_cv_std'], capsize=4,
           label='Especializado (CV)', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['origen'])
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Modelo unificado vs especializados por plataforma')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    grafico = Path(args.salida) / 'comparativa.png'
    plt.savefig(grafico, dpi=150)
    plt.close()
    print(f"[✓] Gráfico:        {grafico}")

    # 5. Guardar índice de modelos en JSON para que app.py los cargue
    indice = {
        'modelo_unificado': args.modelo_unificado,
        'modelos_especializados': {
            row['origen']: f"especializados/modelo_{row['origen']}.pkl"
            for _, row in df_res.iterrows()
        }
    }
    with open(Path(args.salida) / 'indice_modelos.json', 'w', encoding='utf-8') as f:
        json.dump(indice, f, indent=2, ensure_ascii=False)
    print(f"[✓] Índice modelos: {Path(args.salida) / 'indice_modelos.json'}")

    print("\n" + "═"*64)
    print(" SIGUIENTE PASO ".center(64, "═"))
    print("═"*64)
    print("  1. Revisa la tabla comparativa y el gráfico.")
    print("  2. Para usar en la app, modifica app.py para enrutar al modelo")
    print("     correcto según origen detectado (ver app_enrutador.py).")


if __name__ == '__main__':
    main()