from .logger import get_logger
from .utils import get_transform, normalize, setup_seed
try:
    from .effecient_metric import Evaluator
except ImportError:
    from .metric import Evaluator as _SklearnEvaluator

    class Evaluator(_SklearnEvaluator):
        def __init__(self, device=None, metrics=None, sample_level=False):
            super().__init__(metrics=metrics or [])

        def run(self, results, cls_name, logger=None):
            converted = {}
            for key, value in results.items():
                if hasattr(value, "detach"):
                    converted[key] = value.detach().cpu().numpy()
                else:
                    converted[key] = value
            return super().run(converted, cls_name, logger)

from .result_saver import SelectedHeatmapSaver, resolve_corruption_save_path, save_class_metrics, save_sample_scores
from .visualization import visualizer
