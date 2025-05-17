# Seleccionar el modelo
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor


def select_model(model_name, **kwargs):
    if model_name == "knn":
        return KNeighborsRegressor(**kwargs)
    elif model_name == "elasticnet":
        return ElasticNet(**kwargs)
    else:
        raise ValueError("Modelo no soportado. Use 'knn' o 'elasticnet'.")
