"""Let tests import `mcls` when pytest runs from mod-classifier/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The repo root, for `pipeline.features` (real_to_iq), used by the DroneRF loader.
sys.path.append(str(Path(__file__).resolve().parent.parent))
