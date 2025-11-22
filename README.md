# Zero-R & One-R Classifiers

Herramientas computacionales para clasificación supervisada mediante algoritmos fundamentales de aprendizaje automático.

---

## Visión General

Esta suite de software implementa dos técnicas esenciales en el campo del análisis predictivo:

**Zero-R (Regla Cero)**  
Modelo de referencia que establece una predicción constante basándose únicamente en la distribución de clases del conjunto de entrenamiento. Su valor radica en proporcionar un umbral mínimo de rendimiento contra el cual medir algoritmos más sofisticados.

**One-R (Una Regla)**  
Sistema de inducción de reglas que identifica el atributo individual con mayor capacidad discriminativa. Construye un clasificador simple pero efectivo mediante la evaluación sistemática de cada característica del dataset.

---

## Capacidades del Sistema

El framework ofrece las siguientes operaciones:

- Parseo automático de datasets tabulares en formato Markdown
- Motor de clasificación Zero-R con análisis de frecuencias
- Generador de reglas One-R con selección óptima de atributos
- Módulo de evaluación comparativa de rendimiento
- Interface de inferencia para clasificación de nuevas observaciones

---

## Dependencias Técnicas

Entorno de ejecución requerido:

```
Python >= 3.x
pandas
collections (biblioteca estándar)
```

---

## Guía de Ejecución

Pasos para ejecutar el clasificador:

**1.** Prepare su conjunto de datos en formato tabla Markdown  
**2.** Configure la ruta del archivo en el punto de entrada principal  
**3.** Lance el módulo mediante:

```bash
python taquitos.py
```

---

## Resultados Esperados

El sistema genera reportes analíticos con la siguiente estructura:

```
Dataset importado: 20 observaciones × 8 variables

[Zero-R]
→ Predicción constante: 'Sí'
→ Exactitud: 65.00%

[One-R]  
→ Atributo seleccionado: 'Estudiante'
→ Exactitud: 100.00%
→ Mapeo de reglas: {'1': 'Sí', '2': 'Sí', '3': 'No', ...}

[Análisis Comparativo]
Mejora de One-R sobre baseline: +35.00 puntos porcentuales

[Validación]
Instancias correctamente clasificadas: 20/20 (100.00%)
```

---

## Arquitectura Modular

Componentes principales del sistema:

| Módulo | Responsabilidad |
|--------|-----------------|
| `cargar_datos()` | Pipeline de ingesta y transformación |
| `zero_r()` | Motor del clasificador de regla cero |
| `one_r()` | Algoritmo de inducción de reglas unitarias |
| `predecir_one_r()` | Motor de inferencia y clasificación |
| `main()` | Orquestador del flujo de ejecución |

---

## Información Académica

**Institución:** Universidad de Guadalajara - CUCEI  
**Curso:** Minería de Datos  
**Actividad:** 5.2 - Algoritmos de Clasificación Baseline  
**Tema:** Implementación y Análisis de Zero-R y One-R
