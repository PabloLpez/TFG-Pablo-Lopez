"""
entrenar_ia5.py
----------------
Versión corregida sobre entrenar_ia4.py:

  1. Limpieza (imputación + clipping) DENTRO del pipeline para que cada fold
     de CV use sus propias estadísticas (sin data leakage).
  2. Leave-One-Platform-Out (LOPO) para cuantificar sesgo de dominio.
  3. Métricas estratificadas por origen en el reporte de test.
  4. Sample_weight aplicado de forma consistente (también en CV opcionalmente).
  5. PESOS_ORIGEN a 1.0 por defecto (con dataset balanceado ya no hace falta).
  6. Calibración isotónica opcional de probabilidades.
  7. Permutation importance calculada sobre el modelo del split (consistente).
"""

import argparse, pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split, LeaveOneGroupOut)
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAY_XGB = True
except ImportError:
    HAY_XGB = False
    print("[!] XGBoost no instalado. Usa: pip install xgboost")

try:
    import shap
    HAY_SHAP = True
except ImportError:
    HAY_SHAP = False


NOMBRES_FEATURES = [
    "face_detectada",
    "face_parpadeo_freq",
    "face_parpadeo_simetria",
    "face_boca_apertura_std",
    "face_landmark_temblor",
    "face_eye_aspect_std",
    "face_simetria_facial",
    "face_borde_estabilidad",
    "face_iluminacion_consist",
]

# Con dataset balanceado por origen ya no hace falta reweighting.
# Se deja la estructura por si se quiere experimentar con --usar_pesos.
PESOS_ORIGEN_DEFAULT = {
    'movil_directo': 1.0, 'desconocido': 1.0, 'youtube': 1.0,
    'twitter': 1.0, 'whatsapp': 1.0, 'instagram': 1.0,
}


# ═══════════════════════════════════════════════════════════════════
#  TRANSFORMADOR DE LIMPIEZA (dentro del pipeline = sin leakage)
# ═══════════════════════════════════════════════════════════════════

class LimpiadorFeatures(BaseEstimator, TransformerMixin):
    """Imputa NaN con mediana y recorta outliers a ±sigma_clip·σ.
    Los parámetros se aprenden SOLO en fit (en CV, cada fold usa sus propios
    estadísticos calculados sobre el train de ese fold)."""

    def __init__(self, sigma_clip=3.0):
        self.sigma_clip = sigma_clip

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(X, axis=0)
        # mean/std tras imputar para que sean estables ante NaN
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


# ═══════════════════════════════════════════════════════════════════
#  CARGA
# ═══════════════════════════════════════════════════════════════════

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)

    # Eliminar columnas one-hot origen_* (eran filtración directa de la plataforma)
    cols_origen = [c for c in df.columns if c.startswith('origen_')]
    if cols_origen:
        df = df.drop(columns=cols_origen)
        print(f"[+] Columnas one-hot eliminadas: {cols_origen}")

    if 'etiqueta' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'etiqueta': 'label'})

    # Garantizar que todas las features existan (NaN si falta, imputa el pipeline)
    for col in NOMBRES_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    n_reales = (df['label'] == 0).sum()
    n_falsos = (df['label'] == 1).sum()
    print(f"\n[+] Dataset: {len(df)} | Reales: {n_reales} | Falsos: {n_falsos}")

    # Diagnóstico de balance por origen (CRUCIAL para diagnosticar sesgo)
    if 'origen' in df.columns:
        print("\n[+] Distribución origen × clase:")
        tabla = pd.crosstab(df['origen'], df['label'])
        if 0 in tabla.columns: tabla = tabla.rename(columns={0: 'real'})
        if 1 in tabla.columns: tabla = tabla.rename(columns={1: 'fake'})
        print(tabla)

        print("\n[+] Avisos de balance:")
        algun_aviso = False
        for o in df['origen'].unique():
            sub = df[df['origen'] == o]
            if len(sub) < 5:
                continue
            r = int((sub['label'] == 0).sum())
            f = int((sub['label'] == 1).sum())
            if min(r, f) == 0:
                print(f"   ⚠ '{o}': clase única ({r}r/{f}f) — LOPO ignorará")
                algun_aviso = True
            elif max(r, f) / max(min(r, f), 1) > 4:
                print(f"   ⚠ '{o}': desbalance extremo {r}r/{f}f — riesgo de sesgo")
                algun_aviso = True
        if not algun_aviso:
            print("   ✓ Sin avisos graves")

    X = df[NOMBRES_FEATURES].values
    y = df['label'].values
    return X, y, NOMBRES_FEATURES, df


def calcular_pesos(df, mapa_pesos):
    """label=1 (IA) → 1.0 siempre. label=0 (real) → según origen."""
    return np.array([
        1.0 if row['label'] == 1
        else mapa_pesos.get(row.get('origen', 'desconocido'), 1.0)
        for _, row in df.iterrows()
    ])


# ═══════════════════════════════════════════════════════════════════
#  MODELOS
# ═══════════════════════════════════════════════════════════════════

def construir_modelos(sigma_clip=3.0, calibrar=False):
    def envolver(clf):
        return CalibratedClassifierCV(clf, method='isotonic', cv=3) if calibrar else clf

    modelos = {
        'rf': Pipeline([
            ('limp', LimpiadorFeatures(sigma_clip)),
            ('clf', envolver(RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight='balanced',
                random_state=42, n_jobs=-1)))]),
        'gbm': Pipeline([
            ('limp', LimpiadorFeatures(sigma_clip)),
            ('clf', envolver(GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                subsample=0.8, random_state=42)))]),
        'lr': Pipeline([
            ('limp', LimpiadorFeatures(sigma_clip)),
            ('scaler', StandardScaler()),
            ('clf', envolver(LogisticRegression(
                C=1.0, class_weight='balanced', max_iter=1000, random_state=42)))]),
    }
    if HAY_XGB:
        modelos['xgb'] = Pipeline([
            ('limp', LimpiadorFeatures(sigma_clip)),
            ('clf', envolver(XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, n_jobs=-1)))])
    return modelos


# ═══════════════════════════════════════════════════════════════════
#  EVALUACIONES
# ═══════════════════════════════════════════════════════════════════

def evaluar_cv(modelos, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}
    print("\n" + "═" * 64)
    print(" CV ESTRATIFICADA 5-FOLD ".center(64, "═"))
    print("═" * 64)
    for nombre, modelo in modelos.items():
        s = cross_validate(modelo, X, y, cv=cv,
                           scoring=['accuracy', 'roc_auc', 'f1'], n_jobs=-1)
        resultados[nombre] = {
            'accuracy': s['test_accuracy'].mean(),
            'acc_std':  s['test_accuracy'].std(),
            'roc_auc':  s['test_roc_auc'].mean(),
            'auc_std':  s['test_roc_auc'].std(),
            'f1':       s['test_f1'].mean(),
        }
        r = resultados[nombre]
        print(f"  [{nombre.upper():4s}]  Acc: {r['accuracy']:.3f}±{r['acc_std']:.3f} | "
              f"AUC: {r['roc_auc']:.3f}±{r['auc_std']:.3f} | F1: {r['f1']:.3f}")
    mejor = max(resultados, key=lambda k: resultados[k]['roc_auc'])
    print(f"\n[✓] Mejor según AUC-CV: {mejor.upper()} ({resultados[mejor]['roc_auc']:.3f})")
    return mejor, resultados


def evaluar_lopo(modelo, X, y, df):
    """Leave-One-Platform-Out: entrena con N-1 orígenes, testa en el restante.
    Si el AUC LOPO ≈ AUC CV → el modelo generaliza entre plataformas.
    Si AUC LOPO << AUC CV → todavía hay sesgo de dominio."""
    if 'origen' not in df.columns:
        print("[!] Sin columna 'origen', LOPO omitido")
        return None

    grupos, nombres = pd.factorize(df['origen'])
    logo = LeaveOneGroupOut()

    print("\n" + "═" * 64)
    print(" LEAVE-ONE-PLATFORM-OUT ".center(64, "═"))
    print("═" * 64)
    print("  (entrena con N-1 plataformas, testa en la 1 restante)\n")

    resultados = []
    for tr_idx, te_idx in logo.split(X, y, grupos):
        plat = nombres[grupos[te_idx[0]]]
        if len(np.unique(y[te_idx])) < 2:
            print(f"  test={plat:18s} | omitido (solo una clase)")
            continue
        if len(te_idx) < 5:
            print(f"  test={plat:18s} | omitido (n<5)")
            continue
        m = deepcopy(modelo)
        m.fit(X[tr_idx], y[tr_idx])
        prob = m.predict_proba(X[te_idx])[:, 1]
        pred = m.predict(X[te_idx])
        auc  = roc_auc_score(y[te_idx], prob)
        acc  = (pred == y[te_idx]).mean()
        f1   = f1_score(y[te_idx], pred, zero_division=0)
        resultados.append({'plat': plat, 'n': len(te_idx),
                           'acc': acc, 'auc': auc, 'f1': f1})
        print(f"  test={plat:18s} | n={len(te_idx):4d} | "
              f"acc={acc:.3f} | AUC={auc:.3f} | F1={f1:.3f}")

    if resultados:
        aucs = [r['auc'] for r in resultados]
        accs = [r['acc'] for r in resultados]
        print(f"\n  → LOPO medio: acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
              f"AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
    return resultados


def reporte_por_origen(df_test, y_te, y_pred, y_prob):
    print("\n  DESGLOSE POR ORIGEN (test set):")
    print(f"  {'origen':18s} {'n':>4s}  {'acc':>5s}  {'prec':>5s}  {'rec':>5s}  {'AUC':>5s}")
    print(f"  {'-'*18} {'-'*4}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for o in sorted(df_test['origen'].unique()):
        mask = (df_test['origen'].values == o)
        if mask.sum() < 3:
            continue
        yo, yp, yr = y_te[mask], y_pred[mask], y_prob[mask]
        acc = (yo == yp).mean()
        try:
            prec = precision_score(yo, yp, zero_division=0)
            rec  = recall_score(yo, yp, zero_division=0)
            auc  = roc_auc_score(yo, yr) if len(np.unique(yo)) > 1 else float('nan')
        except Exception:
            prec = rec = auc = float('nan')
        print(f"  {o:18s} {mask.sum():>4d}  {acc:.3f}  {prec:.3f}  {rec:.3f}  {auc:.3f}")


# ═══════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO FINAL (split 80/20 para reporte e importancia)
# ═══════════════════════════════════════════════════════════════════

def entrenar_final(modelo, X, y, df, nombres, nombre_modelo,
                   usar_pesos=False, mapa_pesos=None):
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    df_te = df.iloc[idx_te].reset_index(drop=True)

    fit_params = {}
    if usar_pesos:
        pesos = calcular_pesos(df.iloc[idx_tr], mapa_pesos)
        fit_params['clf__sample_weight'] = pesos

    try:
        modelo.fit(X_tr, y_tr, **fit_params)
    except (TypeError, ValueError) as e:
        print(f"[!] sample_weight no soportado ({e.__class__.__name__}), entrenando sin pesos")
        modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)[:, 1]

    print("\n" + "═" * 64)
    print(" REPORTE TEST 20% ".center(64, "═"))
    print("═" * 64)
    print(classification_report(y_te, y_pred, target_names=['Real', 'IA/Falso']))
    print(f"  Confusión:\n{confusion_matrix(y_te, y_pred)}")
    print(f"  ROC-AUC global: {roc_auc_score(y_te, y_prob):.4f}")

    if 'origen' in df_te.columns:
        reporte_por_origen(df_te, y_te, y_pred, y_prob)

    # Importancia de features
    clf_inner = modelo.named_steps['clf']
    if hasattr(clf_inner, 'feature_importances_'):
        imp = clf_inner.feature_importances_
        tipo = 'nativa'
    else:
        imp = permutation_importance(modelo, X_te, y_te,
                                     n_repeats=10, random_state=42,
                                     n_jobs=-1).importances_mean
        tipo = 'permutation'
    imp = imp / (imp.sum() + 1e-10)
    df_imp = pd.DataFrame({'feature': nombres, 'importancia': imp}) \
                .sort_values('importancia', ascending=False)

    print(f"\n  TOP 15 FEATURES (importancia {tipo}):")
    for _, r in df_imp.head(15).iterrows():
        barra = '█' * int(r['importancia'] * 200)
        print(f"    {r['feature']:32s} {r['importancia']:.4f}  {barra}")

    # Detección automática de features "sospechosas" (proxy de plataforma)
    if 'origen' in df.columns:
        print("\n  ANÁLISIS DE PROXIES DE PLATAFORMA:")
        df['es_red'] = df['origen'].isin(['twitter', 'instagram', 'whatsapp']).astype(int)
        sospechosas = []
        for f in nombres:
            v = df[f].dropna()
            if len(v) < 10 or v.std() == 0:
                continue
            c = df[f].corr(df['es_red'])
            if abs(c) > 0.4:
                imp_f = df_imp[df_imp['feature'] == f]['importancia'].values[0]
                sospechosas.append((f, c, imp_f))
        if sospechosas:
            print(f"    Features con |corr(es_red)| > 0.4 (revisa si son útiles o atajos):")
            for f, c, i in sorted(sospechosas, key=lambda x: -abs(x[1])):
                print(f"      {f:32s}  corr={c:+.3f}  importancia={i:.3f}")
        else:
            print("    ✓ No se detectan correlaciones fuertes con plataforma")

    # Gráfico de importancias
    fig, ax = plt.subplots(figsize=(10, 7))
    df_imp.head(15).plot.barh(x='feature', y='importancia', ax=ax, color='steelblue')
    ax.set_title(f'Importancia — {nombre_modelo.upper()} ({tipo})')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('importancia_features.png', dpi=150)
    plt.close()
    print("\n  [✓] importancia_features.png")

    # SHAP (solo con árboles)
    if HAY_SHAP and hasattr(clf_inner, 'feature_importances_'):
        try:
            X_te_clean = modelo.named_steps['limp'].transform(X_te)
            sv = shap.TreeExplainer(clf_inner).shap_values(X_te_clean)
            shap.summary_plot(sv[1] if isinstance(sv, list) else sv,
                              X_te_clean, feature_names=nombres, show=False)
            plt.tight_layout()
            plt.savefig('shap_resumen.png', dpi=150)
            plt.close()
            print("  [✓] shap_resumen.png")
        except Exception as e:
            print(f"  [!] SHAP falló: {e}")

    return modelo, df_imp


def guardar_modelo(modelo, nombres, resultados_cv, resultados_lopo, mejor, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump({
            'modelo': modelo,
            'feature_names': nombres,
            'cv_resultados': resultados_cv,
            'lopo_resultados': resultados_lopo,
            'best_model_name': mejor,
        }, f)
    print(f"\n[✓] Modelo guardado: {ruta}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',         default='features.csv')
    ap.add_argument('--modelo',      default='all', help="rf | gbm | lr | xgb | all")
    ap.add_argument('--salida',      default='modelo.pkl')
    ap.add_argument('--sigma_clip',  type=float, default=3.0,
                    help="Recorte de outliers (default 3σ, antes 5σ)")
    ap.add_argument('--calibrar',    action='store_true',
                    help="Calibrar probabilidades con isotónica")
    ap.add_argument('--usar_pesos',  action='store_true',
                    help="Aplicar pesos por origen (no recomendado tras balanceo)")
    ap.add_argument('--saltar_lopo', action='store_true')
    ap.add_argument('--excluir_features', nargs='+', default=[],
                    help="Lista de features a excluir del modelo (separadas por espacios)")
    args = ap.parse_args()

    X, y, nombres, df = cargar_datos(args.csv)
    if len(X) < 20:
        print("[!] Muy pocos vídeos (<20).")
        return

    # Excluir features si se ha indicado
    if args.excluir_features:
        idx_keep = [i for i, n in enumerate(nombres) if n not in args.excluir_features]
        excluidas = [n for n in nombres if n in args.excluir_features]
        no_encontradas = [n for n in args.excluir_features if n not in nombres]
        if excluidas:
            print(f"\n[+] Excluyendo {len(excluidas)} features: {excluidas}")
        if no_encontradas:
            print(f"[!] No encontradas (revisa el nombre): {no_encontradas}")
        X = X[:, idx_keep]
        nombres = [nombres[i] for i in idx_keep]
        print(f"[+] Features restantes: {len(nombres)}")

    modelos = construir_modelos(sigma_clip=args.sigma_clip, calibrar=args.calibrar)
    if args.modelo != 'all' and args.modelo in modelos:
        modelos = {args.modelo: modelos[args.modelo]}

    # 1. CV estratificada → elegir mejor modelo
    mejor, resultados = evaluar_cv(modelos, X, y)

    # 2. LOPO sobre el mejor → diagnosticar sesgo de dominio
    lopo = None
    if not args.saltar_lopo:
        lopo = evaluar_lopo(modelos[mejor], X, y, df)
        if lopo:
            auc_cv   = resultados[mejor]['roc_auc']
            auc_lopo = np.mean([r['auc'] for r in lopo])
            print(f"\n[DIAGNÓSTICO DE SESGO]")
            print(f"  AUC CV estratificada: {auc_cv:.3f}")
            print(f"  AUC LOPO medio:       {auc_lopo:.3f}")
            print(f"  Caída: {(auc_cv - auc_lopo):+.3f}")
            if auc_cv - auc_lopo > 0.10:
                print(f"  ⚠ Caída > 0.10 → todavía hay sesgo de dominio significativo")
            else:
                print(f"  ✓ Caída < 0.10 → el modelo generaliza razonablemente entre plataformas")

    # 3. Train/test 80-20 → reporte final, importancia, métricas por origen
    modelo_final, _ = entrenar_final(
        modelos[mejor], X, y, df, nombres, mejor,
        usar_pesos=args.usar_pesos, mapa_pesos=PESOS_ORIGEN_DEFAULT)

    # 4. Reentrenar con TODO para producción y guardar
    print("\n[+] Reentrenando con todos los datos para producción...")
    fit_params = {}
    if args.usar_pesos:
        pesos = calcular_pesos(df, PESOS_ORIGEN_DEFAULT)
        fit_params['clf__sample_weight'] = pesos
    try:
        modelo_final.fit(X, y, **fit_params)
    except (TypeError, ValueError):
        modelo_final.fit(X, y)

    guardar_modelo(modelo_final, nombres, resultados, lopo, mejor, args.salida)


if __name__ == '__main__':
    main()