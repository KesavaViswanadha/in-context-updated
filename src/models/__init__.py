from core import ContextModel
from .transformer import TransformerModel
from .mamba_no_attention import MambaNoAttentionModel
from .mambafirstformer import MambaFirstGPT2TransformerModel
from .mambaformer import MambaformerModel
from .mod_transformers import ModTransformerModel
from .linear import (
    LeastSquaresModel,
    AveragingModel,
    LassoModel,
)
from .gradient_mlp import GDModel
from .nearest_neighbors import KNNModel
from .xgboost import XGBoostModel
from .decision_tree import DecisionTreeModel

MODELS: dict[str, type[ContextModel]] = {
    "gpt2" : TransformerModel,

    "least squares" : LeastSquaresModel,
    "knn"           : KNNModel,
    "averaging"     : AveragingModel,
    "lasso"         : LassoModel,
    "grad mlp"      : GDModel,
    "decision tree" : DecisionTreeModel,
    "xgboost"       : XGBoostModel,
    "mambafirstgpt2"       : MambaFirstGPT2TransformerModel,
    "mambaonly"     : MambaNoAttentionModel,
    "mambaformer_classic"   : MambaformerModel,
    "mod_transformer"     : ModTransformerModel
}

__all__ = [
    "MODELS"
]
