"""
=============================================================================
CLASIFICADOR DE ESTADO DE SALUD MEDIANTE MACHINE LEARNING
=============================================================================
CE en Desarrollo de aplicaciones en lenguaje Python
Proyecto Final Intermodular - Proyecto 3

Autores: [Tu nombre]
Fecha:   2025 - 2026
Dataset: Pima Indians Diabetes Dataset (descargado automáticamente)
=============================================================================

Descripción:
    Este script implementa el ciclo completo de un proyecto de Machine Learning:
    1. Ingesta de datos (CSV)
    2. Preprocesamiento (limpieza, nulos, normalización)
    3. Análisis Exploratorio (EDA) con gráficas
    4. Entrenamiento del modelo (80% train / 20% test)
    5. Predicción y evaluación (accuracy, reporte completo)

Librerías:
    - numpy, pandas   → manipulación de datos
    - matplotlib      → visualización
    - scikit-learn    → modelado ML
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTACIONES
# ---------------------------------------------------------------------------
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # backend sin pantalla (para guardar figuras)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# 1. CONSTANTES Y CONFIGURACIÓN
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
#DATASET_PATH = "diabetes.csv"
#OUTPUT_DIR = "graficas"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "diabetes.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "graficas")
RANDOM_STATE = 42
TEST_SIZE = 0.20       # 80% entrenamiento / 20% prueba

# Columnas del dataset Pima Indians Diabetes
COLUMNS = [
    "Embarazos",
    "Glucosa",
    "PresionArterial",
    "GrosorPiel",
    "Insulina",
    "IMC",
    "FuncionDiabetes",
    "Edad",
    "Resultado",            # 0 = sin diabetes, 1 = con diabetes
]

# Valores que representan "dato ausente" en este dataset (codificados como 0)
ZERO_AS_NULL_COLS = [
    "Glucosa",
    "PresionArterial",
    "GrosorPiel",
    "Insulina",
    "IMC",
]


# ---------------------------------------------------------------------------
# 2. UTILIDADES
# ---------------------------------------------------------------------------
def separador(titulo: str) -> None:
    """Imprime un separador visual en consola para estructurar la salida."""
    ancho = 70
    print("\n" + "=" * ancho)
    print(f"  {titulo}")
    print("=" * ancho)


def crear_directorio(ruta: str) -> None:
    """Crea el directorio de salida si no existe."""
    os.makedirs(ruta, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. INGESTA DE DATOS
# ---------------------------------------------------------------------------
def descargar_dataset(url: str, destino: str) -> None:
    """
    Descarga el dataset desde una URL si no existe localmente.
    Si la descarga falla, genera un dataset sintético equivalente.

    Parámetros
    ----------
    url     : str  — URL del archivo CSV
    destino : str  — ruta local donde se guardará
    """
    if os.path.exists(destino):
        print(f"  Dataset encontrado localmente: {destino}")
        return

    try:
        print(f"  Descargando dataset desde:\n  {url}")
        urllib.request.urlretrieve(url, destino)
        print(f"  Dataset guardado en: {destino}")
    except Exception as e:
        print(f"  ⚠ No se pudo descargar ({e}). Usando dataset CSV local.")
        if not os.path.exists(destino):
            raise FileNotFoundError(
                f"Coloca el archivo '{destino}' en el mismo directorio que este script.\n"
                "Descárgalo desde: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
            )


def cargar_datos(ruta: str, columnas: list) -> pd.DataFrame:
    """
    Carga el CSV y asigna nombres de columnas en español.

    Parámetros
    ----------
    ruta     : str   — ruta al archivo CSV
    columnas : list  — nombres de las columnas

    Retorna
    -------
    pd.DataFrame con los datos cargados
    """
    df = pd.read_csv(ruta, header=None, names=columnas)
    print(f"  Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 4. PREPROCESAMIENTO
# ---------------------------------------------------------------------------
def preprocesar(df: pd.DataFrame, cols_nulos: list) -> pd.DataFrame:
    """
    Limpia y preprocesa el DataFrame:
      - Reemplaza ceros inválidos por NaN en columnas biomédicas
      - Imputa los NaN con la mediana de cada columna
      - Verifica y corrige tipos de datos

    Parámetros
    ----------
    df        : pd.DataFrame — datos originales
    cols_nulos: list         — columnas donde 0 significa dato ausente

    Retorna
    -------
    pd.DataFrame limpio y listo para el modelado
    """
    df_clean = df.copy()

    # Reemplazar 0 por NaN en columnas biomédicas
    df_clean[cols_nulos] = df_clean[cols_nulos].replace(0, np.nan)

    print("\n  Valores nulos detectados por columna:")
    nulos = df_clean.isnull().sum()
    print(nulos[nulos > 0].to_string())

    # Imputar con la mediana (robusta frente a outliers)
    for col in cols_nulos:
        mediana = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(mediana)
        print(f"  → '{col}' imputada con mediana = {mediana:.2f}")

    # Asegurar tipos numéricos correctos
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print(f"\n  Valores nulos tras imputación: {df_clean.isnull().sum().sum()}")
    return df_clean


# ---------------------------------------------------------------------------
# 5. ANÁLISIS EXPLORATORIO (EDA)
# ---------------------------------------------------------------------------
def eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Genera y guarda tres visualizaciones de análisis exploratorio:
      1. Distribuciones de variables clave por clase (Resultado)
      2. Diagrama de dispersión Glucosa vs Edad coloreado por clase
      3. Matriz de correlación (heatmap)

    Parámetros
    ----------
    df         : pd.DataFrame — datos limpios
    output_dir : str          — directorio donde se guardan las imágenes
    """
    crear_directorio(output_dir)
    colores = {0: "#2196F3", 1: "#F44336"}   # azul = sano, rojo = diabetes

    # -- Figura 1: Distribuciones por clase -----------------------------------
    variables = ["Glucosa", "IMC", "Edad", "PresionArterial"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Distribución de variables biomédicas por clase\n(Azul = Sin diabetes | Rojo = Con diabetes)",
        fontsize=13, fontweight="bold"
    )
    for ax, var in zip(axes.flatten(), variables):
        for clase, color in colores.items():
            datos = df[df["Resultado"] == clase][var]
            ax.hist(datos, bins=20, alpha=0.6, color=color,
                    label=f"{'Diabetes' if clase else 'Sano'} (n={len(datos)})")
        ax.set_title(var, fontweight="bold")
        ax.set_xlabel(var)
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    ruta1 = os.path.join(output_dir, "01_distribuciones.png")
    plt.savefig(ruta1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfica guardada: {ruta1}")

    # -- Figura 2: Dispersión Glucosa vs Edad ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for clase, color in colores.items():
        subset = df[df["Resultado"] == clase]
        ax.scatter(
            subset["Edad"], subset["Glucosa"],
            c=color, alpha=0.5, s=30, edgecolors="none",
            label=f"{'Diabetes' if clase else 'Sin diabetes'}"
        )
    ax.set_title("Relación entre Edad y Glucosa", fontsize=13, fontweight="bold")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Nivel de Glucosa (mg/dL)")
    ax.legend()
    ax.grid(alpha=0.3)
    ruta2 = os.path.join(output_dir, "02_dispersion_glucosa_edad.png")
    plt.savefig(ruta2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfica guardada: {ruta2}")

    # -- Figura 3: Matriz de correlación --------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    etiquetas = corr.columns.tolist()
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(etiquetas, fontsize=9)
    # Anotar valores en cada celda
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Matriz de correlación entre variables", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta3 = os.path.join(output_dir, "03_matriz_correlacion.png")
    plt.savefig(ruta3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfica guardada: {ruta3}")


# ---------------------------------------------------------------------------
# 6. ENTRENAMIENTO Y EVALUACIÓN
# ---------------------------------------------------------------------------
def entrenar_y_evaluar(df: pd.DataFrame, output_dir: str) -> None:
    """
    Divide los datos (80/20), normaliza, entrena dos modelos
    (Árbol de Decisión y KNN) y muestra las métricas de evaluación.
    También guarda la matriz de confusión del mejor modelo.

    Parámetros
    ----------
    df         : pd.DataFrame — datos limpios
    output_dir : str          — directorio para guardar la matriz de confusión
    """
    # Separar características (X) de la variable objetivo (y)
    X = df.drop(columns=["Resultado"])
    y = df["Resultado"]

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y        # mantener proporción de clases
    )
    print(f"\n  Muestras entrenamiento : {len(X_train)}")
    print(f"  Muestras prueba        : {len(X_test)}")

    # Normalización (StandardScaler: media 0, desviación estándar 1)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Definir modelos a comparar
    modelos = {
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors (k=7)": KNeighborsClassifier(
            n_neighbors=7,
            metric="euclidean"
        ),
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        # Validación cruzada (5 folds) para estimar rendimiento real
        cv_scores = cross_val_score(modelo, X_train_sc, y_train, cv=5, scoring="accuracy")

        # Entrenamiento final sobre el 80%
        modelo.fit(X_train_sc, y_train)

        # Predicción sobre el 20% de prueba
        y_pred = modelo.predict(X_test_sc)
        acc = accuracy_score(y_test, y_pred)

        resultados[nombre] = {
            "modelo": modelo,
            "y_pred": y_pred,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"\n  ─── {nombre} ───")
        print(f"  Accuracy en test           : {acc * 100:.2f}%")
        print(f"  CV accuracy (5-fold)       : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
        print(f"\n  Reporte de clasificación:")
        print(classification_report(y_test, y_pred,
                                    target_names=["Sin diabetes", "Con diabetes"]))

    # Seleccionar el mejor modelo (mayor accuracy en test)
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["accuracy"])
    mejor = resultados[mejor_nombre]
    print(f"\n  ★ Mejor modelo: {mejor_nombre} ({mejor['accuracy'] * 100:.2f}%)")

    # Guardar matriz de confusión del mejor modelo
    cm = confusion_matrix(y_test, mejor["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Sin diabetes", "Con diabetes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {mejor_nombre}", fontweight="bold")
    ruta_cm = os.path.join(output_dir, "04_matriz_confusion.png")
    plt.tight_layout()
    plt.savefig(ruta_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfica guardada: {ruta_cm}")


# ---------------------------------------------------------------------------
# 7. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------
def main() -> None:
    """Orquesta el pipeline completo del proyecto ML."""

    # ── Hito 1: Ingesta de datos ──────────────────────────────────────────
    separador("HITO 1 — INGESTA DE DATOS")
    descargar_dataset(DATASET_URL, DATASET_PATH)
    df_raw = cargar_datos(DATASET_PATH, COLUMNS)
    print("\n  Primeras filas del dataset:")
    print(df_raw.head().to_string())
    print(f"\n  Resumen estadístico:")
    print(df_raw.describe().round(2).to_string())

    # ── Hito 2: Preprocesamiento + EDA ────────────────────────────────────
    separador("HITO 2 — PREPROCESAMIENTO")
    df_clean = preprocesar(df_raw, ZERO_AS_NULL_COLS)

    separador("HITO 2 — ANÁLISIS EXPLORATORIO (EDA)")
    print("  Generando gráficas...")
    eda(df_clean, OUTPUT_DIR)

    # ── Hito 3: Modelado y validación ─────────────────────────────────────
    separador("HITO 3 — ENTRENAMIENTO Y EVALUACIÓN DEL MODELO")
    entrenar_y_evaluar(df_clean, OUTPUT_DIR)

    separador("PIPELINE COMPLETADO ✓")
    print(f"  Gráficas exportadas en la carpeta: {OUTPUT_DIR}/")
    print("  Hito 4: sube el código y las gráficas a GitHub antes de la defensa.\n")


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
