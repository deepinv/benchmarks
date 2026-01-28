from benchopt import BaseSolver

import torch
import deepinv as dinv
from pathlib import Path
from hashlib import md5
from benchopt.utils.class_property import classproperty


class Solver(BaseSolver):
    """Class used to wrap a model as a solver for benchopt.

    To use this class, set the `model` and `name` attributes to run
    it in the model.
    """

    name = "runner-solver"
    model = None
    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        # Make sure the model is on the correct device and has a device
        # attribute to we can properly move the test data later.
        self.model.to(device)
        self.model.device = device

    @classproperty
    def _module_filename(self):
        return Path(__file__).parent / "dynamic_run"

    @classproperty
    def _file_hash(self):
        hasher = md5()
        hasher.update(str(self.model).encode())
        # If you want to be able to call this with models at different training
        # steps, you might want to include the model state_dict in the hash.
        return hasher.hexdigest()

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
