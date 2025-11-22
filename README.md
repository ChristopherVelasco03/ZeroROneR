# 🍺 Clasificadores Zero-R y One-R

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pandas](https://img.shields.io/badge/Pandas-Required-orange.svg)](https://pandas.pydata.org/)

Implementación educativa de algoritmos fundamentales de clasificación supervisada para análisis de datos y aprendizaje automático.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Documentación Detallada](#-documentación-detallada)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Ejemplo de Salida](#-ejemplo-de-salida)
- [Algoritmos Implementados](#-algoritmos-implementados)

---

## 🎯 Descripción

Esta librería implementa dos algoritmos clásicos de clasificación que sirven como línea base (baseline) en proyectos de aprendizaje automático:

### **Zero-R (Regla Cero)**
Clasificador que predice siempre la clase más frecuente en el conjunto de entrenamiento. Aunque simple, establece el rendimiento mínimo que cualquier modelo inteligente debe superar.

### **One-R (Una Regla)**
Algoritmo que selecciona el atributo individual más predictivo y genera reglas de clasificación basadas en sus valores. A pesar de su simplicidad, puede lograr alta precisión en ciertos problemas.

**Caso de Uso:** Análisis de preferencias de cervezas basado en características demográficas y contextuales.

---

## ✨ Características

- ✅ **Carga Inteligente de Datos**: Soporte para CSV y tablas Markdown con auto-detección de formato
- ✅ **Arquitectura Orientada a Objetos**: Diseño modular con clases base abstractas
- ✅ **Evaluación Robusta**: Sistema de validación iterativa con múltiples particiones train/test
- ✅ **Análisis Estadístico**: Cálculo de precisión promedio y desviación estándar
- ✅ **Comparación Automática**: Framework para comparar rendimiento entre modelos
- ✅ **Reproducibilidad**: Control de semillas aleatorias para experimentos replicables
- ✅ **Interfaz Interactiva**: Entrada de parámetros por consola para experimentación
- ✅ **Visualización de Reglas**: Inspección de las reglas generadas por One-R

---

## 🔧 Requisitos

```
Python >= 3.7
pandas >= 1.0.0
numpy >= 1.18.0
```

**Nota:** Las librerías `collections` y `abc` son parte de la biblioteca estándar de Python.

---

## 📥 Instalación

### Opción 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/cervezas-zeror-oner.git
cd cervezas-zeror-oner
```

### Opción 2: Descargar ZIP

Descarga el archivo ZIP desde GitHub y extráelo en tu directorio de trabajo.

### Instalar Dependencias

```bash
pip install pandas numpy
```

O usando requirements.txt (si lo creas):

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso Rápido

### Ejecución Básica

```bash
python cervezas.py
```

El programa te solicitará:
- **Número de iteraciones** para evaluación (ej: 10)
- **Porcentaje de datos** para entrenamiento (ej: 0.7 para 70%)

### Ejemplo de Interacción

```
Ingrese el número de iteraciones para evaluación (ej: 10): 10
Ingrese el porcentaje de datos para entrenamiento (ej: 0.7): 0.7
```

---

## 📖 Documentación Detallada

### Módulos Principales

#### 1️⃣ **Módulo de Carga de Datos**

```python
from cervezas import cargar_datos, dividir_datos

# Cargar datos
datos = cargar_datos('cervezas.txt')

# Preparar características y objetivo
X = datos.drop('Prefiere', axis=1)
y = datos['Prefiere']

# Dividir datos
X_train, X_test, y_train, y_test = dividir_datos(X, y, porcentaje_entrenamiento=0.7, semilla=42)
```

#### 2️⃣ **Módulo de Modelos**

##### Zero-R

```python
from cervezas import ZeroR

# Crear y entrenar modelo
modelo_zeror = ZeroR()
modelo_zeror.entrenar(X_train, y_train)

# Realizar predicciones
predicciones = modelo_zeror.predecir(X_test)

# Evaluar
precision = modelo_zeror.evaluar(X_test, y_test)
print(f"Precisión: {precision:.2%}")
```

##### One-R

```python
from cervezas import OneR

# Crear y entrenar modelo
modelo_oner = OneR()
modelo_oner.entrenar(X_train, y_train)

# Realizar predicciones
predicciones = modelo_oner.predecir(X_test)

# Obtener reglas
reglas = modelo_oner.obtener_reglas()
print(f"Atributo seleccionado: {reglas['atributo']}")
print(f"Reglas: {reglas['reglas']}")
```

#### 3️⃣ **Módulo de Evaluación**

```python
from cervezas import Evaluador

# Evaluación iterativa
resultados = Evaluador.evaluar_modelo_iterativo(
    modelo_clase=OneR,
    X=X,
    y=y,
    num_iteraciones=10,
    porcentaje_entrenamiento=0.7,
    semilla_base=42
)

# Comparar modelos
comparacion = Evaluador.comparar_modelos({
    'Zero-R': resultados_zeror,
    'One-R': resultados_oner
})
```

---

## 📁 Estructura del Proyecto

```
Cervezas_ZeroRule_OneRule/
│
├── cervezas.py          # Código principal con todos los módulos
├── cervezas.txt         # Dataset de ejemplo (preferencias de cervezas)
├── README.md            # Este archivo
└── __pycache__/         # Archivos compilados de Python
```

---

## 📊 Ejemplo de Salida

```
======================================================================
 LIBRERÍA DE CLASIFICACIÓN ZERO-R Y ONE-R
======================================================================

[PASO 1] Cargando datos...
✓ Datos cargados: 20 instancias, 8 columnas
  Columnas: ['Género', 'Edad', 'Ocupación', 'Estudiante', 'Situación_Sentimental', 
             'Clima', 'Música', 'Prefiere']

[PASO 2] Preparando datos...
✓ Características: ['Género', 'Edad', 'Ocupación', 'Estudiante', 
                    'Situación_Sentimental', 'Clima', 'Música']
✓ Variable objetivo: 'Prefiere'
✓ Distribución de clases: {'Clara': 13, 'Oscura': 7}

======================================================================
RESULTADOS DEL MODELO: Zero-R
======================================================================

Iteraciones realizadas: 10

Resultados por iteración:
Iter   Precisión Train    Precisión Test    
---------------------------------------------
1      0.6429             0.6667            
2      0.6429             0.6667            
...

Estadísticas Finales:
  Precisión promedio (entrenamiento): 0.6429
  Precisión promedio (prueba):        0.6667
  Desviación estándar (prueba):       0.0000

======================================================================
RESULTADOS DEL MODELO: One-R
======================================================================

Iteraciones realizadas: 10

Resultados por iteración:
Iter   Precisión Train    Precisión Test    
---------------------------------------------
1      0.8571             0.8333            
2      0.8571             0.8333            
...

Estadísticas Finales:
  Precisión promedio (entrenamiento): 0.8571
  Precisión promedio (prueba):        0.8333
  Desviación estándar (prueba):       0.0000

[Ejemplo de Reglas One-R - Última iteración]
  Atributo seleccionado: 'Clima'
  Reglas generadas: 3 reglas
    Si Clima = Soleado → Clase = Clara
    Si Clima = Nublado → Clase = Clara
    Si Clima = Lluvia → Clase = Oscura

======================================================================
 COMPARACIÓN FINAL
======================================================================

Modelo          Precisión Promedio   Desviación     
--------------------------------------------------
Zero-R          0.6667               0.0000         
One-R           0.8333               0.0000         

CONCLUSIÓN:
  → El modelo One-R tiene mejor rendimiento
  → Diferencia en precisión: 0.1667 (16.67%)

======================================================================
 EVALUACIÓN COMPLETADA
======================================================================
```

---

## 🧮 Algoritmos Implementados

### Zero-R (ZR)

**Principio:** Predecir la clase mayoritaria.

**Ventajas:**
- Extremadamente simple y rápido
- No requiere características
- Establece baseline mínimo

**Desventajas:**
- No aprende patrones
- Ignora todas las características
- Bajo rendimiento en datasets balanceados

**Complejidad Temporal:** $O(n)$ donde $n$ es el número de instancias

---

### One-R (1R)

**Principio:** Crear reglas basadas en el mejor atributo individual.

**Algoritmo:**
1. Para cada atributo:
   - Para cada valor del atributo: asignar la clase más frecuente
   - Contar errores de clasificación
2. Seleccionar el atributo con menor tasa de error
3. Usar sus reglas para clasificación

**Ventajas:**
- Simple e interpretable
- A menudo competitivo con algoritmos complejos
- Genera reglas comprensibles

**Desventajas:**
- Solo usa un atributo
- No captura interacciones entre variables
- Sensible a atributos con muchos valores

**Complejidad Temporal:** $O(m \cdot n)$ donde $m$ = atributos, $n$ = instancias

---

