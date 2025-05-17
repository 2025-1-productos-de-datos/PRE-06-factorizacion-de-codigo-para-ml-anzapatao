#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better
from homework.src._internals.select_model import select_model

x_train, x_test, y_train, y_test = prepare_data(file_path="data/winequality-red.csv")

# Seleccionar el modelo
model = select_model("knn", n_neighbors=5)

model.fit(x_train, y_train)

mse, mae, r2 = calculate_metrics(model, x_train, y_train)
print_metrics("Entrenamiento", mse, mae, r2)

# Metricas de error durante testing
mse, mae, r2 = calculate_metrics(model, x_test, y_test)
print_metrics("Testing", mse, mae, r2)

save_model_if_better(model, x_test, y_test)
