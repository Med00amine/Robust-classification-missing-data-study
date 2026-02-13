from src.models.random_forest import RandomForestModel
from src.models.Knn import KNNModel
from src.models.mlp import MLPModel
from src.models.CNN import CNNModel
from src.models.transformer_encoder import TransformerModel


def get_model(model_name, input_len):

    if model_name == "RandomForest":
        return RandomForestModel()

    elif model_name == "KNN":
        return KNNModel()

    elif model_name == "MLP":
        return MLPModel()

    elif model_name == "CNN":
        return CNNModel()

    elif model_name == "Transformer":
        return TransformerModel(input_len=input_len)

    else:
        raise ValueError("Unknown model name")
