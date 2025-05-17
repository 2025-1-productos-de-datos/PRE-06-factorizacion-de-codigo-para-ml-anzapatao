#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# importacion de librerias
import pandas as pd

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better
from homework.src._internals.select_model import select_model

# descarga de datos
x_train, x_test, y_train, y_test = prepare_data(file_path="data/winequality-red.csv")

# Seleccionar el modelo
model = select_model("elasticnet", alpha=0.5, l1_ratio=0.5, random_state=12345)

model.fit(x_train, y_train)

print()
print(model, ":", sep="")

# Metricas de error durante entrenamiento
mse, mae, r2 = calculate_metrics(model, x_train, y_train)

print()
print_metrics("Entrenamiento", mse, mae, r2)

# Metricas de error durante testing
print()
print("Metricas de testing:")
mse, mae, r2 = calculate_metrics(model, x_test, y_test)
print_metrics("Testing", mse, mae, r2)
save_model_if_better(model, x_test, y_test)
