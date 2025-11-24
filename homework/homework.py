# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import json
import gzip
import os
import pickle
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def _leer_csv_desde_zip(ruta_zip: str, nombre_archivo: str) -> pd.DataFrame:
    with zipfile.ZipFile(ruta_zip, "r") as archivo_zip:
        with archivo_zip.open(nombre_archivo) as archivo_interno:
            return pd.read_csv(archivo_interno)


def _limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_limpio = df.copy()

    # Eliminar ID y renombrar columna objetivo
    df_limpio = df_limpio.drop("ID", axis=1)
    df_limpio = df_limpio.rename(columns={"default payment next month": "default"})

    # Eliminar filas con datos faltantes
    df_limpio = df_limpio.dropna()

    # Remover registros sin información válida en EDUCATION o MARRIAGE
    df_limpio = df_limpio[(df_limpio["EDUCATION"] != 0) & (df_limpio["MARRIAGE"] != 0)]

    # Agrupar niveles de educación > 4 en "others" (4)
    df_limpio.loc[df_limpio["EDUCATION"] > 4, "EDUCATION"] = 4

    return df_limpio


def _construir_busqueda_cv() -> GridSearchCV:
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", StandardScaler(), columnas_numericas),
        ],
        remainder="passthrough",
    )

    pipeline_modelo = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
            ("pca", PCA()),  # todas las componentes
            ("selector_kbest", SelectKBest(score_func=f_classif)),
            ("clasificador", SVC(kernel="rbf", random_state=42)),
        ]
    )

    rejilla_parametros = {
        "pca__n_components": [20, 21],
        "selector_kbest__k": [12],
        "clasificador__kernel": ["rbf"],
        "clasificador__gamma": [0.099],
    }

    return GridSearchCV(
        estimator=pipeline_modelo,
        param_grid=rejilla_parametros,
        cv=10,
        refit=True,
        verbose=1,
        return_train_score=False,
        scoring="balanced_accuracy",
    )


def _registro_metricas(nombre_conjunto: str, y_real, y_predicho) -> dict:
    return {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": precision_score(y_real, y_predicho),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predicho),
        "recall": recall_score(y_real, y_predicho),
        "f1_score": f1_score(y_real, y_predicho),
    }


def _registro_matriz_confusion(nombre_conjunto: str, y_real, y_predicho) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_real, y_predicho).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def _guardar_modelo_comprimido(modelo) -> None:
    Path("files/models").mkdir(parents=True, exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as archivo_modelo:
        pickle.dump(modelo, archivo_modelo)


def _guardar_registros_jsonl(registros: list[dict]) -> None:
    Path("files/output").mkdir(parents=True, exist_ok=True)
    ruta_salida = "files/output/metrics.json"
    with open(ruta_salida, "w", encoding="utf-8") as archivo_salida:
        for registro in registros:
            archivo_salida.write(json.dumps(registro) + "\n")


if __name__ == "__main__":
    ruta_test_zip = "files/input/test_data.csv.zip"
    ruta_train_zip = "files/input/train_data.csv.zip"
    nombre_interno_test = "test_default_of_credit_card_clients.csv"
    nombre_interno_train = "train_default_of_credit_card_clients.csv"

    df_test = _limpiar_dataset(
        _leer_csv_desde_zip(ruta_test_zip, nombre_interno_test)
    )
    df_train = _limpiar_dataset(
        _leer_csv_desde_zip(ruta_train_zip, nombre_interno_train)
    )

    X_train, y_train = df_train.drop("default", axis=1), df_train["default"]
    X_test, y_test = df_test.drop("default", axis=1), df_test["default"]

    busqueda_cv = _construir_busqueda_cv()
    busqueda_cv.fit(X_train, y_train)

    _guardar_modelo_comprimido(busqueda_cv)

    y_train_pred = busqueda_cv.predict(X_train)
    y_test_pred = busqueda_cv.predict(X_test)

    metricas_train = _registro_metricas("train", y_train, y_train_pred)
    metricas_test = _registro_metricas("test", y_test, y_test_pred)

    cm_train = _registro_matriz_confusion("train", y_train, y_train_pred)
    cm_test = _registro_matriz_confusion("test", y_test, y_test_pred)

    _guardar_registros_jsonl(
        [metricas_train, metricas_test, cm_train, cm_test]
    )