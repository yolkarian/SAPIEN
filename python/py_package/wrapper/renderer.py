from ..pysapien.render import SapienRenderer as _SapienRenderer
from ..pysapien.render import RenderMaterial
from warnings import warn


class SapienRenderer(_SapienRenderer):
    def __init__(self, **args):
        warn("SapienRenderer is no no longer needed.", DeprecationWarning, stacklevel=2)
        super().__init__()

    def create_material(self):
        return RenderMaterial()
