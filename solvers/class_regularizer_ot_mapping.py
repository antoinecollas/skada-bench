from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import ClassRegularizerOTMappingAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from xgboost import XGBClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'class_regularizer_ot_mapping'

    requirements = [
        'pip:git+https://github.com/scikit-adaptation/skada.git',
        "pip:xgboost",
        "pip:POT",
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    param_grid = {
        'classregularizerotmappingadapter__reg_e': [1.0, 0.1],
        'classregularizerotmappingadapter__reg_cl': [0.1, 0.01],
        'classregularizerotmappingadapter__norm': ["lpl1", "l1l2"],
        'classregularizerotmappingadapter__metric': ["sqeuclidean"],
        'classregularizerotmappingadapter__max_iter': [10, 100],
        'classregularizerotmappingadapter__max_inner_iter': [100],
        'classregularizerotmappingadapter__tol': [10e-9, 10e-10],
    }

    def get_estimator(self):
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            ClassRegularizerOTMappingAdapter(),
            XGBClassifier()
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True),
        )
