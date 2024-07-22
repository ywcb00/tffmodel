from tffmodel.ModelBuilderUtils import getMetrics

import numpy as np
from prettytable import PrettyTable

class ModelUtils:
    @classmethod
    def printEvaluations(self_class, eval_dict, config, first_col_name="Model"):
        metric_objects = getMetrics(config)
        eval_table = PrettyTable([first_col_name] + [mo.name for mo in metric_objects])
        for model_name, metrics in eval_dict.items():
            eval_table.add_row(
                [model_name] + [metrics[mo.name] for mo in metric_objects])
        return eval_table

    @classmethod
    def averageModelWeights(self_class, model_weights):
        result = [np.average([mw[layer_idx] for mw in model_weights], axis=0) for layer_idx in range(len(model_weights[0]))]
        return result
