"""
enrutador_modelos.py
---------------------
Módulo de enrutamiento: dado el origen detectado, devuelve la ruta del
modelo apropiado. Si el origen no tiene modelo especializado, usa el
unificado como fallback.

Uso desde app.py:
    from enrutador_modelos import seleccionar_modelo
    ruta = seleccionar_modelo(origen)
    a.cargar_modelo(ruta)
"""

import os, json


def cargar_indice(ruta_indice='especializados/indice_modelos.json'):
    """Carga el índice de modelos. Si no existe, devuelve None."""
    if not os.path.exists(ruta_indice):
        return None
    with open(ruta_indice, 'r', encoding='utf-8') as f:
        return json.load(f)


def seleccionar_modelo(origen, ruta_indice='especializados/indice_modelos.json',
                       fallback='modelo_v2.pkl'):
    """
    Devuelve la ruta del modelo más apropiado para el origen detectado.
    
    Lógica:
      - Si existe modelo especializado para ese origen → lo usa.
      - Si no, devuelve el modelo unificado (fallback).
    
    Args:
        origen: nombre del origen detectado (instagram, twitter, etc.)
        ruta_indice: ruta del JSON con índice de modelos
        fallback: ruta del modelo unificado por defecto
    
    Returns:
        tupla (ruta_modelo, tipo)
        tipo es 'especializado' o 'unificado'
    """
    indice = cargar_indice(ruta_indice)

    if indice is None:
        return fallback, 'unificado'

    especializados = indice.get('modelos_especializados', {})

    if origen in especializados:
        ruta = especializados[origen]
        if os.path.exists(ruta):
            return ruta, 'especializado'

    # Fallback
    unificado = indice.get('modelo_unificado', fallback)
    return unificado, 'unificado'


def listar_modelos_disponibles(ruta_indice='especializados/indice_modelos.json'):
    """Devuelve un dict con los orígenes que tienen modelo especializado."""
    indice = cargar_indice(ruta_indice)
    if indice is None:
        return {}
    return {
        origen: ruta
        for origen, ruta in indice.get('modelos_especializados', {}).items()
        if os.path.exists(ruta)
    }


if __name__ == '__main__':
    # Pequeño test
    print("Modelos disponibles:")
    disponibles = listar_modelos_disponibles()
    for origen, ruta in disponibles.items():
        print(f"  {origen:20s} → {ruta}")

    print("\nPruebas de enrutamiento:")
    for origen in ['instagram', 'twitter', 'movil_directo', 'youtube', 'desconocido']:
        ruta, tipo = seleccionar_modelo(origen)
        print(f"  {origen:20s} → {ruta} ({tipo})")
