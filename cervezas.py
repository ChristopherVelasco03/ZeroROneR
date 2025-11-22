import pandas as pd
import numpy as np
from collections import Counter
from abc import ABC, abstractmethod


# =====================================================================
# MÓDULO DE CARGA DE DATOS
# =====================================================================

def cargar_datos(ruta_archivo, separador=None, columna_objetivo=None):
    """
    Carga datos desde un archivo CSV o formato tabla markdown.
    
    Parámetros de entrada:
    - ruta_archivo (str): Ruta al archivo de datos
    - separador (str, opcional): Separador del CSV (',' por defecto, None para auto-detectar)
    - columna_objetivo (str, opcional): Nombre de la columna objetivo/clase
    
    Parámetros de salida:
    - DataFrame de pandas con los datos cargados
    
    Función: Carga y prepara datos desde archivo para análisis
    """
    try:
        # Intentar cargar como CSV estándar
        if separador is None:
            # Auto-detectar separador
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                primera_linea = f.readline()
                if '|' in primera_linea and primera_linea.count('|') > 2:
                    # Formato tabla markdown
                    return _cargar_markdown(ruta_archivo)
                elif ',' in primera_linea:
                    separador = ','
                elif '\t' in primera_linea:
                    separador = '\t'
                else:
                    separador = ','
        
        df = pd.read_csv(ruta_archivo, sep=separador, encoding='utf-8')
        return df
    
    except Exception as e:
        # Si falla, intentar formato markdown
        return _cargar_markdown(ruta_archivo)


def _cargar_markdown(ruta_archivo):
    """
    Carga datos desde un archivo en formato tabla markdown.
    
    Parámetros de entrada:
    - ruta_archivo (str): Ruta al archivo markdown
    
    Parámetros de salida:
    - DataFrame de pandas con los datos
    
    Función: Procesa formato de tabla markdown y convierte a DataFrame
    """
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    lineas = contenido.strip().split('\n')
    cabeceras = [h.strip() for h in lineas[0].split('|') if h.strip()]
    lineas_datos = lineas[2:]
    
    datos = []
    for linea in lineas_datos:
        valores = [val.strip() for val in linea.split('|') if val.strip()]
        if valores:
            datos.append(valores)
    
    df = pd.DataFrame(datos, columns=cabeceras)
    
    # Convertir columnas numéricas
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    return df


def dividir_datos(X, y, porcentaje_entrenamiento=0.7, semilla=None):
    """
    Divide datos en conjuntos de entrenamiento y prueba con selección aleatoria uniforme.
    
    Parámetros de entrada:
    - X (DataFrame): Características del conjunto de datos
    - y (Series): Variable objetivo/clase
    - porcentaje_entrenamiento (float): Proporción de datos para entrenamiento (0.0 a 1.0)
    - semilla (int, opcional): Semilla para reproducibilidad
    
    Parámetros de salida:
    - Tupla (X_train, X_test, y_train, y_test): Conjuntos divididos
    
    Función: Realiza división aleatoria estratificada de datos
    """
    if semilla is not None:
        np.random.seed(semilla)
    
    # Crear índices aleatorios
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Calcular punto de división
    punto_division = int(len(X) * porcentaje_entrenamiento)
    
    # Dividir índices
    indices_train = indices[:punto_division]
    indices_test = indices[punto_division:]
    
    # Crear conjuntos
    X_train = X.iloc[indices_train].reset_index(drop=True)
    X_test = X.iloc[indices_test].reset_index(drop=True)
    y_train = y.iloc[indices_train].reset_index(drop=True)
    y_test = y.iloc[indices_test].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


# =====================================================================
# MÓDULO DE MODELOS (Clases Base)
# =====================================================================

class ClasificadorBase(ABC):
    """
    Clase abstracta base para clasificadores.
    Define la interfaz común para todos los modelos.
    """
    
    def __init__(self):
        self.entrenado = False
    
    @abstractmethod
    def entrenar(self, X, y):
        """Entrena el modelo con datos de entrenamiento"""
        pass
    
    @abstractmethod
    def predecir(self, X):
        """Realiza predicciones sobre nuevos datos"""
        pass
    
    def evaluar(self, X, y):
        """
        Evalúa el modelo en un conjunto de datos.
        
        Parámetros de entrada:
        - X (DataFrame): Características
        - y (Series): Etiquetas verdaderas
        
        Parámetros de salida:
        - float: Precisión del modelo (accuracy)
        
        Función: Calcula métricas de rendimiento del modelo
        """
        predicciones = self.predecir(X)
        correctos = sum(1 for pred, real in zip(predicciones, y) if pred == real)
        precision = correctos / len(y) if len(y) > 0 else 0
        return precision


class ZeroR(ClasificadorBase):
    """
    Implementación del algoritmo Zero-R.
    Predice siempre la clase más frecuente.
    """
    
    def __init__(self):
        super().__init__()
        self.clase_mayoritaria = None
    
    def entrenar(self, X, y):
        """
        Entrena el modelo Zero-R.
        
        Parámetros de entrada:
        - X (DataFrame): Características (no se usan en Zero-R)
        - y (Series): Etiquetas de clase
        
        Parámetros de salida:
        - None (modifica el estado interno del objeto)
        
        Función: Determina la clase mayoritaria en los datos de entrenamiento
        """
        contador = Counter(y)
        self.clase_mayoritaria = contador.most_common(1)[0][0]
        self.entrenado = True
    
    def predecir(self, X):
        """
        Predice clases usando Zero-R.
        
        Parámetros de entrada:
        - X (DataFrame): Características a predecir
        
        Parámetros de salida:
        - list: Lista de predicciones (todas iguales a la clase mayoritaria)
        
        Función: Retorna la clase mayoritaria para todas las instancias
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero")
        return [self.clase_mayoritaria] * len(X)


class OneR(ClasificadorBase):
    """
    Implementación del algoritmo One-R.
    Selecciona el mejor atributo único para clasificación.
    """
    
    def __init__(self):
        super().__init__()
        self.mejor_atributo = None
        self.reglas = {}
        self.clase_por_defecto = None
    
    def entrenar(self, X, y):
        """
        Entrena el modelo One-R.
        
        Parámetros de entrada:
        - X (DataFrame): Características del conjunto de entrenamiento
        - y (Series): Etiquetas de clase
        
        Parámetros de salida:
        - None (modifica el estado interno del objeto)
        
        Función: Encuentra el mejor atributo y genera reglas de clasificación
        """
        mejor_precision = 0
        
        for atributo in X.columns:
            reglas = {}
            errores = 0
            
            for valor in X[atributo].unique():
                indices = X[atributo] == valor
                clases_valor = y[indices]
                
                if len(clases_valor) > 0:
                    contador = Counter(clases_valor)
                    clase_mayoritaria = contador.most_common(1)[0][0]
                    reglas[valor] = clase_mayoritaria
                    
                    errores += sum(1 for i, etiqueta in enumerate(y) 
                                  if indices.iloc[i] and etiqueta != clase_mayoritaria)
            
            precision = 1 - (errores / len(y)) if len(y) > 0 else 0
            
            if precision > mejor_precision:
                self.mejor_atributo = atributo
                mejor_precision = precision
                self.reglas = reglas
        
        # Establecer clase por defecto
        contador = Counter(y)
        self.clase_por_defecto = contador.most_common(1)[0][0]
        self.entrenado = True
    
    def predecir(self, X):
        """
        Predice clases usando One-R.
        
        Parámetros de entrada:
        - X (DataFrame): Características a predecir
        
        Parámetros de salida:
        - list: Lista de predicciones basadas en las reglas
        
        Función: Aplica las reglas generadas para clasificar nuevas instancias
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero")
        
        predicciones = []
        for _, fila in X.iterrows():
            valor = fila[self.mejor_atributo]
            if valor in self.reglas:
                predicciones.append(self.reglas[valor])
            else:
                predicciones.append(self.clase_por_defecto)
        
        return predicciones
    
    def obtener_reglas(self):
        """
        Obtiene las reglas generadas por el modelo.
        
        Parámetros de entrada:
        - None
        
        Parámetros de salida:
        - dict: Diccionario con el atributo seleccionado y sus reglas
        
        Función: Retorna información sobre las reglas de clasificación
        """
        return {
            'atributo': self.mejor_atributo,
            'reglas': self.reglas
        }


# =====================================================================
# MÓDULO DE EVALUACIÓN
# =====================================================================

class Evaluador:
    """
    Clase para evaluar modelos con múltiples iteraciones.
    """
    
    @staticmethod
    def evaluar_modelo_iterativo(modelo_clase, X, y, num_iteraciones=10, 
                                 porcentaje_entrenamiento=0.7, semilla_base=42):
        """
        Evalúa un modelo con múltiples iteraciones de train/test split.
        
        Parámetros de entrada:
        - modelo_clase (class): Clase del modelo a evaluar (ZeroR o OneR)
        - X (DataFrame): Características completas
        - y (Series): Etiquetas completas
        - num_iteraciones (int): Número de iteraciones a realizar
        - porcentaje_entrenamiento (float): Proporción de datos para entrenamiento
        - semilla_base (int): Semilla base para reproducibilidad
        
        Parámetros de salida:
        - dict: Diccionario con resultados de todas las iteraciones
        
        Función: Realiza evaluación robusta mediante múltiples particiones aleatorias
        """
        resultados = {
            'iteraciones': [],
            'precision_entrenamiento': [],
            'precision_prueba': []
        }
        
        for i in range(num_iteraciones):
            # Crear modelo nuevo para cada iteración
            modelo = modelo_clase()
            
            # Dividir datos con semilla diferente para cada iteración
            semilla = semilla_base + i if semilla_base is not None else None
            X_train, X_test, y_train, y_test = dividir_datos(
                X, y, porcentaje_entrenamiento, semilla
            )
            
            # Entrenar y evaluar
            modelo.entrenar(X_train, y_train)
            precision_train = modelo.evaluar(X_train, y_train)
            precision_test = modelo.evaluar(X_test, y_test)
            
            resultados['iteraciones'].append(i + 1)
            resultados['precision_entrenamiento'].append(precision_train)
            resultados['precision_prueba'].append(precision_test)
        
        # Calcular estadísticas
        resultados['precision_promedio_entrenamiento'] = np.mean(resultados['precision_entrenamiento'])
        resultados['precision_promedio_prueba'] = np.mean(resultados['precision_prueba'])
        resultados['desviacion_prueba'] = np.std(resultados['precision_prueba'])
        
        return resultados
    
    @staticmethod
    def comparar_modelos(resultados_modelos):
        """
        Compara resultados de múltiples modelos.
        
        Parámetros de entrada:
        - resultados_modelos (dict): Diccionario con resultados de cada modelo
        
        Parámetros de salida:
        - dict: Comparación y estadísticas de los modelos
        
        Función: Analiza y compara el rendimiento de diferentes modelos
        """
        comparacion = {}
        
        for nombre, resultados in resultados_modelos.items():
            comparacion[nombre] = {
                'precision_promedio': resultados['precision_promedio_prueba'],
                'desviacion': resultados['desviacion_prueba']
            }
        
        return comparacion



# =====================================================================
# MÓDULO PRINCIPAL (Ejemplo de Uso)
# =====================================================================

def imprimir_resultados_iteracion(nombre_modelo, resultados):
    """
    Imprime los resultados de evaluación iterativa de un modelo.
    
    Parámetros de entrada:
    - nombre_modelo (str): Nombre del modelo evaluado
    - resultados (dict): Diccionario con resultados de iteraciones
    
    Parámetros de salida:
    - None (imprime en consola)
    
    Función: Formatea y muestra resultados de evaluación
    """
    print(f"\n{'='*70}")
    print(f"RESULTADOS DEL MODELO: {nombre_modelo}")
    print(f"{'='*70}")
    
    print(f"\nIteraciones realizadas: {len(resultados['iteraciones'])}")
    print(f"\nResultados por iteración:")
    print(f"{'Iter':<6} {'Precisión Train':<18} {'Precisión Test':<18}")
    print("-" * 45)
    
    for i, (train, test) in enumerate(zip(resultados['precision_entrenamiento'], 
                                          resultados['precision_prueba'])):
        print(f"{i+1:<6} {train:<18.4f} {test:<18.4f}")
    
    print(f"\n{'Estadísticas Finales:'}")
    print(f"  Precisión promedio (entrenamiento): {resultados['precision_promedio_entrenamiento']:.4f}")
    print(f"  Precisión promedio (prueba):        {resultados['precision_promedio_prueba']:.4f}")
    print(f"  Desviación estándar (prueba):       {resultados['desviacion_prueba']:.4f}")


def main():
    """
    Función principal que demuestra el uso de la librería.
    
    Parámetros de entrada:
    - None (usa configuración por defecto, puede modificarse)
    
    Parámetros de salida:
    - None (ejecuta demostración completa)
    
    Función: Ejemplo completo de carga, entrenamiento y evaluación de modelos
    """
    print("\n" + "="*70)
    print(" LIBRERÍA DE CLASIFICACIÓN ZERO-R Y ONE-R")
    print("="*70)
    
    # ========== PASO 1: CARGAR DATOS ==========
    print("\n[PASO 1] Cargando datos...")
    archivo = 'cervezas.txt'
    datos = cargar_datos(archivo)
    print(f"✓ Datos cargados: {datos.shape[0]} instancias, {datos.shape[1]} columnas")
    print(f"  Columnas: {list(datos.columns)}")
    
    # ========== PASO 2: PREPARAR DATOS ==========
    print("\n[PASO 2] Preparando datos...")
    columna_objetivo = 'Prefiere'
    X = datos.drop(columna_objetivo, axis=1)
    y = datos[columna_objetivo]
    print(f"✓ Características: {list(X.columns)}")
    print(f"✓ Variable objetivo: '{columna_objetivo}'")
    print(f"✓ Distribución de clases: {dict(Counter(y))}")
    
    # ========== PASO 3: CONFIGURAR EVALUACIÓN ==========
    print("\n[PASO 3] Configurando evaluación...")
    num_iteraciones = int(input("Ingrese el número de iteraciones para evaluación (ej: 10): ") or "10")
    porcentaje_train = float(input("Ingrese el porcentaje de datos para entrenamiento (ej: 0.7): ") or "0.7")
    print(f"✓ Número de iteraciones: {num_iteraciones}")
    print(f"✓ División: {porcentaje_train*100:.0f}% entrenamiento, {(1-porcentaje_train)*100:.0f}% prueba")
    
    # ========== PASO 4: EVALUAR ZERO-R ==========
    print("\n[PASO 4] Evaluando modelo Zero-R...")
    resultados_zeror = Evaluador.evaluar_modelo_iterativo(
        ZeroR, X, y, 
        num_iteraciones=num_iteraciones,
        porcentaje_entrenamiento=porcentaje_train,
        semilla_base=42
    )
    imprimir_resultados_iteracion("Zero-R", resultados_zeror)
    
    # ========== PASO 5: EVALUAR ONE-R ==========
    print("\n[PASO 5] Evaluando modelo One-R...")
    resultados_oner = Evaluador.evaluar_modelo_iterativo(
        OneR, X, y,
        num_iteraciones=num_iteraciones,
        porcentaje_entrenamiento=porcentaje_train,
        semilla_base=42
    )
    imprimir_resultados_iteracion("One-R", resultados_oner)
    
    # Mostrar reglas de One-R (última iteración como ejemplo)
    print("\n[Ejemplo de Reglas One-R - Última iteración]")
    modelo_ejemplo = OneR()
    X_train, X_test, y_train, y_test = dividir_datos(X, y, porcentaje_train, 42)
    modelo_ejemplo.entrenar(X_train, y_train)
    reglas = modelo_ejemplo.obtener_reglas()
    print(f"  Atributo seleccionado: '{reglas['atributo']}'")
    print(f"  Reglas generadas: {len(reglas['reglas'])} reglas")
    if len(reglas['reglas']) <= 10:
        for valor, clase in list(reglas['reglas'].items())[:10]:
            print(f"    Si {reglas['atributo']} = {valor} → Clase = {clase}")
    
    # ========== PASO 6: COMPARAR MODELOS ==========
    print("\n[PASO 6] Comparación de modelos...")
    print(f"\n{'='*70}")
    print(" COMPARACIÓN FINAL")
    print(f"{'='*70}")
    
    comparacion = Evaluador.comparar_modelos({
        'Zero-R': resultados_zeror,
        'One-R': resultados_oner
    })
    
    print(f"\n{'Modelo':<15} {'Precisión Promedio':<20} {'Desviación':<15}")
    print("-" * 50)
    for nombre, stats in comparacion.items():
        print(f"{nombre:<15} {stats['precision_promedio']:<20.4f} {stats['desviacion']:<15.4f}")
    
    # Determinar mejor modelo
    mejor_modelo = max(comparacion.items(), key=lambda x: x[1]['precision_promedio'])
    diferencia = abs(comparacion['Zero-R']['precision_promedio'] - 
                    comparacion['One-R']['precision_promedio'])
    
    print(f"\n{'CONCLUSIÓN:'}")
    print(f"  → El modelo {mejor_modelo[0]} tiene mejor rendimiento")
    print(f"  → Diferencia en precisión: {diferencia:.4f} ({diferencia*100:.2f}%)")
    
    print("\n" + "="*70)
    print(" EVALUACIÓN COMPLETADA")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()