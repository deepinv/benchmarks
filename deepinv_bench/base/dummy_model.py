from deepinv.models import Reconstructor


class DummyModel(Reconstructor):
    def __call__(self, x, physics=None):
        return physics.A_adjoint(x)
