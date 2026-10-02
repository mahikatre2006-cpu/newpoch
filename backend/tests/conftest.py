import os

# Set before app.config loads backend/.env, so default test runs can never touch Neon.
os.environ["DATABASE_URL"] = ""


class NullElements:
    """Element engine stand-in for tests that are not about M4: only the background, instantly."""

    status = {"loaded": ["null"], "unavailable": {}}

    def detect(self, ctx, layers=None):
        from app.elements import resolve

        return resolve([], ctx.image.shape[:2]), []


class NullScanpath:
    """No scanpath model: the pipeline uses the winner-take-all path on the saliency map."""

    model = None
    status = {"loaded": [], "unavailable": {}}


class FixedScanpath:
    """A scanpath 'model' that visits the given frame points, for testing the wiring."""

    status = {"loaded": ["fixed"], "unavailable": {}}
    model = object()

    def __init__(self, points):
        self.points = points

    def predict(self, ctx):
        return list(self.points)
