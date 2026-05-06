# 🩺 Clasificador de Estado de Salud mediante Machine Learning

> Proyecto Final Intermodular — CE en Desarrollo de Aplicaciones en Lenguaje Python  
> IES Font de Sant Lluís · Curso 2025–2026

---

## 📋 Descripción

Aplicación de Machine Learning que predice si un paciente tiene riesgo de diabetes a partir de indicadores biomédicos. Implementa el ciclo completo de un proyecto de datos: ingesta, limpieza, análisis exploratorio, entrenamiento y evaluación de modelos.

**Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 pacientes, 8 variables predictoras, variable objetivo binaria (0 = sin diabetes / 1 = con diabetes).

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/SalvaIES/clasificador-salud-ml.git
cd clasificador-salud-ml
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Ejecutar

```bash
python clasificador_salud.py
```

El script descarga el dataset automáticamente. Si no hay conexión, coloca `diabetes.csv` en la misma carpeta.

---

## 🗂️ Estructura del repositorio

```
clasificador-salud-ml/
│
├── clasificador_salud.py       # Script principal (pipeline completo)
├── diabetes.csv                # Dataset original
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
│
├── graficas/                   # Generadas automáticamente al ejecutar
│   ├── 01_distribuciones.png
│   ├── 02_dispersion_glucosa_edad.png
│   ├── 03_matriz_correlacion.png
│   └── 04_matriz_confusion.png
│
└── docs/
    └── Memoria_Tecnica_Proyecto3_ML.docx
```

---

## 🔬 Pipeline ML

```
CSV  →  Preprocesamiento  →  EDA  →  Train/Test Split  →  Modelo  →  Métricas
```

| Etapa | Descripción |
|---|---|
| **Ingesta** | Carga del CSV con `pandas`, asignación de columnas |
| **Preprocesamiento** | Sustitución de ceros inválidos por `NaN`, imputación con mediana |
| **EDA** | 3 gráficas: distribuciones, dispersión Glucosa/Edad, matriz de correlación |
| **División** | 80% entrenamiento / 20% prueba con `stratify` |
| **Normalización** | `StandardScaler` ajustado solo sobre el train |
| **Modelos** | Árbol de Decisión (`max_depth=5`) y KNN (`k=7`) |
| **Evaluación** | Accuracy, precision, recall, F1-score + validación cruzada 5-fold |

---

## 📊 Resultados

| Modelo | Accuracy (test) | CV Accuracy (5-fold) |
|---|---|---|
| Árbol de Decisión | ~81% | ~84% ± 1.6% |
| K-Nearest Neighbors (k=7) | ~85% | ~88% ± 2.5% |

> **Mejor modelo:** K-Nearest Neighbors con k=7

---

## 📦 Librerías utilizadas

| Librería | Uso |
|---|---|
| `numpy` | Operaciones numéricas y arrays |
| `pandas` | Carga, limpieza y manipulación de datos |
| `matplotlib` | Visualización de gráficas |
| `scikit-learn` | Modelos ML, normalización y métricas |

---

## 📁 Variables del dataset

| Variable | Descripción |
|---|---|
| Embarazos | Número de embarazos previos |
| Glucosa | Concentración de glucosa en plasma (mg/dL) |
| PresionArterial | Presión arterial diastólica (mm Hg) |
| GrosorPiel | Grosor del pliegue cutáneo del tríceps (mm) |
| Insulina | Insulina sérica a las 2 horas (mu U/ml) |
| IMC | Índice de Masa Corporal (kg/m²) |
| FuncionDiabetes | Función pedigrí de diabetes |
| Edad | Edad del paciente (años) |
| **Resultado** | **0 = Sin diabetes / 1 = Con diabetes** |

---

## 🗓️ Planificación (Scrum)

| Hito | Semana | Descripción |
|---|---|---|
| Hito 1 | Sem. 1 | Selección del dataset, product backlog, repositorio GitHub |
| Hito 2 | Sem. 2 | Ingesta, limpieza de datos y EDA (3 gráficas) |
| Hito 3 | Sem. 3 | Modelado, validación cruzada, revisión PEP8 |
| Hito 4 | Sem. 4 | Entrega, memoria técnica y defensa oral |

---

## 👤 Autor

**Salvador Martínez Bolinches**  
CE en Desarrollo de Aplicaciones en Lenguaje Python  
IES Font de Sant Lluís · 2025–2026

---

## 📄 Licencia

Proyecto académico — uso educativo.
