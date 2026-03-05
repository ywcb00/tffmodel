from tffmodel.ModelBuilderUtils import getMetrics

from enum import Enum

class ModelType(Enum):
    KERAS = 1
    PYTORCH = 2

class ModelUtils:
    @classmethod
    def getModelObject(self_class, config):
        # TODO: this implementation creates a model object for every call
        match config["modeltype_id"]:
            case ModelType.KERAS:
                from tffmodel.KerasModel import KerasModel
                return KerasModel(config)
            case ModelType.PYTORCH:
                from tffmodel.PyTorchModel import PyTorchModel
                return PyTorchModel(config)
            case _:
                raise NotImplementedError

    @classmethod
    def getModelClass(self_class, config):
        # TODO: this implementation creates a model object for every call
        match config["modeltype_id"]:
            case ModelType.KERAS:
                from tffmodel.KerasModel import KerasModel
                return KerasModel
            case ModelType.PYTORCH:
                from tffmodel.PyTorchModel import PyTorchModel
                return PyTorchModel
            case _:
                raise NotImplementedError

    @classmethod
    def printEvaluations(self_class, eval_dict, config, first_col_name="Model"):
        from prettytable import PrettyTable

        metric_objects = getMetrics(config)
        eval_table = PrettyTable([first_col_name] + [mo.name for mo in metric_objects])
        for model_name, metrics in eval_dict.items():
            eval_table.add_row(
                [model_name] + [metrics[mo.name] for mo in metric_objects])
        return eval_table
