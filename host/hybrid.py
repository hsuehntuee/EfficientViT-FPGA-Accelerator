"""Shared pieces for run_image.py / eval_dataset.py / bench.py.

HybridModel: EfficientViT-Seg-B1 where the four Stage 4 blocks are executed by a backend
(FPGA over XDMA, or the integer reference model) and everything else runs in FP32 on the CPU.
"""
import glob
import importlib.abc
import importlib.machinery
import os
import sys
import time
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from xdma_run import BRAM_MAP, WEIGHT_FILES, REG_START, REG_DONE, REG_CYCLES  # noqa: E402

A_FRAC, W_FRAC = 5, 11
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DATASETS = {"cityscapes": {"n_classes": 19, "ckpt": "b1-cityscapes.pt"},
            "ade20k": {"n_classes": 150, "ckpt": "efficientvit_seg_b1_ade20k.pt"}}   # same backbone, only the head differs


def default_ckpt(gdir, dataset):
    return os.path.join(gdir, "assets", DATASETS[dataset]["ckpt"])


def load_label(path, dataset):
    """Label png -> int array with 255 = ignore. Cityscapes: trainIds as stored. ADE20K: 0 = other -> ignore, 1..150 -> 0..149."""
    from PIL import Image
    lab = np.array(Image.open(path)).astype(np.int64)
    if lab.ndim == 3:
        lab = lab[..., 0]
    if dataset == "ade20k":
        lab = np.where(lab == 0, 255, lab - 1)
    return lab


def label_for(img_path, dataset):
    """Label path from the image path in the standard layouts (Cityscapes gtFine, ADE20K annotations/)."""
    if dataset == "ade20k":
        p = img_path.replace("/images/", "/annotations/").replace(os.sep + "images" + os.sep, os.sep + "annotations" + os.sep)
        return os.path.splitext(p)[0] + ".png"
    return img_path.replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")


def palette(n_cls):
    if n_cls == 19:
        import evit_hw_golden as g
        return g.CITYSCAPES_PALETTE
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, size=(n_cls, 3)).astype(np.uint8)


def overlay(img_rgb, mask, n_cls):
    from PIL import Image
    h, w = img_rgb.shape[:2]
    m = np.array(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST))
    color = palette(n_cls)[np.clip(m, 0, n_cls - 1)]
    return (0.4 * img_rgb + 0.6 * color).astype(np.uint8)


# ---------------------------------------------------------------- imports
def _stub_optional_packages():
    """efficientvit imports onnx/triton/... at module level; stub them when absent (never used here)."""
    import torch  # noqa: F401  real torch before the stubs
    roots = ("onnx", "onnxsim", "torchprofile", "triton", "segment_anything")

    class Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] not in roots:
                return None
            try:
                if importlib.machinery.PathFinder.find_spec(name, path) is not None:
                    return None
            except Exception:
                pass
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            m = mock.MagicMock()
            m.__spec__, m.__path__, m.__name__ = spec, [], spec.name
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, Stub())


def load_golden_module(golden_repo=None):
    """Import evit_hw_golden.py (software-side integer model). Returns (module, its directory)."""
    cands = [golden_repo, os.environ.get("EVIT_HW_GOLDEN")]
    cands += [os.path.join(REPO, "model", "evit_hw_golden")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "evit_hw_golden.py")):
            sys.path.insert(0, d)
            _stub_optional_packages()
            os.environ.setdefault("EFFICIENTVIT_ROOT", os.path.join(REPO, "third_party", "efficientvit"))
            import evit_hw_golden as g
            return g, d
    sys.exit("evit_hw_golden.py not found: pass --golden-repo or set EVIT_HW_GOLDEN")


# ---------------------------------------------------------------- XDMA
class Xdma:
    """Raw access: BRAM writes/reads through the DMA channels, registers through the user BAR."""

    def __init__(self, user="/dev/xdma0_user", h2c="/dev/xdma0_h2c_0", c2h="/dev/xdma0_c2h_0"):
        import mmap
        import struct
        self.struct = struct
        self.fd_user = os.open(user, os.O_RDWR | os.O_SYNC)
        self.regs = mmap.mmap(self.fd_user, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.h2c = os.open(h2c, os.O_WRONLY)
        self.c2h = os.open(c2h, os.O_RDONLY)

    def wr(self, off, v):
        self.regs.seek(off)
        self.regs.write(self.struct.pack("<I", v))

    def rd(self, off):
        self.regs.seek(off)
        return self.struct.unpack("<I", self.regs.read(4))[0]

    def write_bram(self, name, data):
        off, size = BRAM_MAP[name]
        if len(data) > size:
            raise ValueError(f"{name}: {len(data)} bytes > {size}")
        done = 0
        while done < len(data):
            n = os.pwrite(self.h2c, data[done:done + 65536], off + done)
            if n <= 0:
                raise OSError(f"XDMA write failed at {name}+{done}")
            done += n

    def read_bram(self, name, nbytes):
        off, size = BRAM_MAP[name]
        out, done = bytearray(), 0
        while done < min(nbytes, size):
            chunk = os.pread(self.c2h, min(65536, nbytes - done), off + done)
            if not chunk:
                break
            out += chunk
            done += len(chunk)
        return bytes(out)

    def write_weights(self, weights):
        for n, bram in WEIGHT_FILES:
            if n != "x":
                self.write_bram(bram, np.ascontiguousarray(weights[n], dtype="<i2").tobytes())

    def run(self, timeout=5.0):
        """Pulse start, poll done, return (cycles, seconds waited)."""
        t0 = time.perf_counter()
        self.wr(REG_START, 0)
        self.wr(REG_START, 1)
        while not (self.rd(REG_DONE) & 1):
            if time.perf_counter() - t0 > timeout:
                self.wr(REG_START, 0)
                raise TimeoutError(f"accelerator did not finish in {timeout}s (cycles so far {self.rd(REG_CYCLES)})")
        t = time.perf_counter() - t0
        cycles = self.rd(REG_CYCLES)
        self.wr(REG_START, 0)
        return int(cycles), t

    def close(self):
        self.regs.close()
        for fd in (self.fd_user, self.h2c, self.c2h):
            os.close(fd)


class FpgaBackend:
    """One block per call: write x + weights, start, poll done, read y."""
    name = "fpga"

    def __init__(self, timeout=5.0):
        self.x, self.timeout = Xdma(), timeout

    def run_block(self, x_int16, weights, **_):
        t0 = time.perf_counter()
        self.x.write_bram("BRAM_X", np.ascontiguousarray(x_int16, dtype="<i2").tobytes())
        self.x.write_weights(weights)
        t1 = time.perf_counter()
        cycles, t_run = self.x.run(self.timeout)
        t2 = time.perf_counter()
        y = np.frombuffer(self.x.read_bram("BRAM_Y", x_int16.size * 2), dtype="<i2").copy()
        t3 = time.perf_counter()
        return y, {"cycles": cycles, "t_upload_ms": 1e3 * (t1 - t0), "t_compute_ms": 1e3 * (t2 - t1), "t_readback_ms": 1e3 * (t3 - t2)}

    def close(self):
        self.x.close()


class FileBackend:
    """Replay recorded hardware outputs: block i returns <y_dir>/block_i_y_hw.bin (as saved by xdma_run.py --save
    or run_batch.py); blocks without a file fall back to the integer model. Lets the software side draw the mask
    the board actually produced without torch on the board host."""
    name = "file"

    def __init__(self, g, y_dir):
        self.g, self.y_dir = g, y_dir
        self.block = 0

    def run_block(self, x_int16, weights, params=None, x_chw=None):
        import os
        t0 = time.perf_counter()
        f = os.path.join(self.y_dir, f"block_{self.block}_y_hw.bin")
        self.block += 1
        if os.path.exists(f):
            y = np.fromfile(f, dtype="<i2")
            src = "board file"
        else:
            y = self.g.format_act(self.g.run_block_int(params, x_chw, W_FRAC, A_FRAC, dw_act=True)["y_golden"])
            src = "model"
        return y, {"cycles": None, "t_upload_ms": 0.0, "t_compute_ms": 1e3 * (time.perf_counter() - t0), "t_readback_ms": 0.0, "source": src}

    def close(self):
        pass


class ModelBackend:
    """Dry run: the integer reference model stands in for the FPGA (its output is what the FPGA must produce)."""
    name = "model"

    def __init__(self, g):
        self.g = g

    def run_block(self, x_int16, weights, params=None, x_chw=None):
        t0 = time.perf_counter()
        y = self.g.format_act(self.g.run_block_int(params, x_chw, W_FRAC, A_FRAC, dw_act=True)["y_golden"])
        return y, {"cycles": None, "t_upload_ms": 0.0, "t_compute_ms": 1e3 * (time.perf_counter() - t0), "t_readback_ms": 0.0}

    def close(self):
        pass


# ---------------------------------------------------------------- model
class HybridModel:
    def __init__(self, g, ckpt, backend, size=512, dataset="cityscapes"):
        import torch
        from torchvision import transforms
        self.g, self.backend, self.size, self.dataset = g, backend, size, dataset
        self.n_classes = DATASETS[dataset]["n_classes"]
        seg_b1, EfficientViTBlock = g.import_efficientvit()
        self.model = seg_b1(dataset=dataset)
        chk = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(chk["state_dict"] if "state_dict" in chk else chk, strict=True)
        self.model.eval()
        self.blocks = [m for n, m in self.model.named_modules() if isinstance(m, EfficientViTBlock) and "stages.3" in n]
        self.params = [g.extract_block_params(b, W_FRAC, A_FRAC) for b in self.blocks]
        self.weights = [g.format_weights(p) for p in self.params]
        self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.mode = "fp32"
        self.cache = {}
        self.per_block = {}
        self.keep_io = False
        self.io = {}
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._hook(i))

    def _hook(self, i):
        import torch

        def f(mod, inp, out):
            if self.mode == "fp32":
                self.cache[("fp32", i)] = out.detach().clone()
                return out
            g = self.g
            x_chw = g.quantize(inp[0][0], A_FRAC)
            x_fmt = g.format_act(x_chw)
            ref_fmt = g.format_act(g.run_block_int(self.params[i], x_chw, W_FRAC, A_FRAC, dw_act=True)["y_golden"])
            y_fmt, info = self.backend.run_block(x_fmt, self.weights[i], params=self.params[i], x_chw=x_chw)
            y_fmt = y_fmt[:ref_fmt.size]
            d = np.abs(y_fmt.astype(np.int64) - ref_fmt.astype(np.int64))
            info.update({"bit_exact": bool((d == 0).all()), "mismatches": int((d != 0).sum()), "max_abs_diff": int(d.max())})
            self.per_block[i] = info
            if self.keep_io:
                self.io[i] = (x_fmt.astype("<i2"), y_fmt.astype("<i2"))
            C, H, W = inp[0].shape[1:]
            y_chw = y_fmt.astype(np.int64).reshape(H * W, C // 16, 16).transpose(1, 2, 0).reshape(C, H, W)
            out = torch.from_numpy(y_chw / float(1 << A_FRAC)).to(out.dtype).unsqueeze(0)
            self.cache[("hw", i)] = out.detach().clone()
            return out
        return f

    def preprocess(self, pil_img):
        return self.transform(pil_img.convert("RGB")).unsqueeze(0)

    def logits(self, x, mode, out_size=None):
        """Run the network in 'fp32' or 'hw' mode; logits upsampled (bilinear) to out_size, default input size."""
        import torch
        import torch.nn.functional as F
        self.mode = mode
        with torch.no_grad():
            o = self.model(x)
        o = o["output"] if isinstance(o, dict) else o
        return F.interpolate(o, size=out_size or (self.size, self.size), mode="bilinear", align_corners=False)

    def block_similarity(self):
        """cosine / MSE of each block's output, hw vs fp32 (needs both modes run on the same input)."""
        import torch.nn.functional as F
        res = {}
        for i in range(len(self.blocks)):
            f, q = self.cache[("fp32", i)].flatten(), self.cache[("hw", i)].flatten()
            res[i] = {"cosine_vs_fp32": F.cosine_similarity(f.unsqueeze(0), q.unsqueeze(0)).item(), "mse_vs_fp32": F.mse_loss(f, q).item()}
        return res


# ---------------------------------------------------------------- metrics
def confusion(pred, label, n_cls=19, ignore=255):
    m = label != ignore
    return np.bincount(n_cls * label[m].astype(np.int64) + pred[m].astype(np.int64), minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def iou_from_confusion(c):
    inter = np.diag(c).astype(np.float64)
    union = c.sum(0) + c.sum(1) - inter
    per_class = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return per_class, float(np.nanmean(per_class)), float(inter.sum() / max(c.sum(), 1))


CITYSCAPES_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
                      "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
