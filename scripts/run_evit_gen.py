"""Run evit_hw_golden.py (software-side golden generator) on a machine where the optional packages the
efficientvit import chain pulls in (onnx, onnxsim, torchprofile, triton, segment_anything) are not installed.
They are stubbed; the integer golden never uses them. Needs torch, torchvision, timm, numpy, pillow.

  python scripts/run_evit_gen.py gen --out <dir> --no-plot
  python scripts/run_evit_gen.py gen --out <dir> --no-plot --no-dw-act
  python scripts/run_evit_gen.py compare <golden> <hw dump> --hw-chunk-major 16 [--golden-part q]

Paths: the efficientvit package is taken from <repo>/third_party/efficientvit, the generator from
<repo>/model/evit_hw_golden/evit_hw_golden.py (or $EVIT_HW_GOLDEN).
"""
import glob
import importlib.abc
import importlib.machinery
import os
import runpy
import sys
from unittest import mock

import torch  # noqa: F401  real torch must be imported before the stubs go in

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB_ROOTS = ("onnx", "onnxsim", "torchprofile", "triton", "segment_anything")


class StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB_ROOTS:
            try:
                if importlib.machinery.PathFinder.find_spec(name, path) is not None:
                    return None          # really installed: use it
            except Exception:
                pass
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock()
        m.__spec__, m.__path__, m.__name__ = spec, [], spec.name
        return m

    def exec_module(self, module):
        pass


_cands = [os.path.join(REPO, "model", "evit_hw_golden", "evit_hw_golden.py")]
gen = os.environ.get("EVIT_HW_GOLDEN") or next((c for c in _cands if os.path.isfile(c)), None)
if not gen:
    sys.exit("evit_hw_golden.py not found: set EVIT_HW_GOLDEN")
sys.meta_path.insert(0, StubFinder())
os.environ.setdefault("EFFICIENTVIT_ROOT", os.path.join(REPO, "third_party", "efficientvit"))
sys.path.insert(0, os.path.dirname(gen))
sys.argv = [gen] + sys.argv[1:]
runpy.run_path(gen, run_name="__main__")
