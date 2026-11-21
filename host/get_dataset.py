"""Fetch a labelled street-scene set in the Cityscapes layout that eval_dataset.py / export_batch.py expect
(--cityscapes): <out>/leftImg8bit/val/<city>/*_leftImg8bit.png + <out>/gtFine/val/<city>/*_gtFine_labelTrainIds.png

  # 1. Cityscapes val (500 images, the dataset the model was trained on). Needs a free account:
  #    https://www.cityscapes-dataset.com/register/   (the site has no anonymous download)
  export CITYSCAPES_USERNAME=... CITYSCAPES_PASSWORD=...
  python host/get_dataset.py cityscapes --out data/cityscapes            # ~11.3 GB download (images zip is 11 GB)
  python host/get_dataset.py cityscapes --out data/cityscapes --keep-train   # also keep train/test (default: val only)

  # 2. KITTI semantic (200 labelled images, Cityscapes label ids, no account, 313 MB)
  python host/get_dataset.py kitti --out data/kitti

Both produce *_labelTrainIds.png (19 classes, 255 = ignore) from the raw labelIds with the official
Cityscapes id -> trainId table, so cityscapesscripts is not needed. Downloads resume if interrupted.
"""
import argparse
import glob
import os
import shutil
import sys
import urllib.request
import zipfile

import numpy as np
from PIL import Image

# Cityscapes labels.py: id -> trainId (everything not listed is 255 / ignore)
ID2TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 31: 16, 32: 17, 33: 18}
LUT = np.full(256, 255, np.uint8)
for k, v in ID2TRAIN.items():
    LUT[k] = v

CS_LOGIN = "https://www.cityscapes-dataset.com/login/"
CS_FILE = "https://www.cityscapes-dataset.com/file-handling/?packageID={}"
CS_PACKAGES = {"gtFine_trainvaltest.zip": 1, "leftImg8bit_trainvaltest.zip": 3}
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"
ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"          # ~923 MB, no account
ADE_CKPT_URL = "https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b1_ade20k.pt"   # official, 42.84 mIoU @512


def human(n):
    return f"{n / 1e9:.2f} GB" if n > 1e9 else f"{n / 1e6:.0f} MB"


def download(url, dst, opener=None, headers=None):
    """Resumable download with progress."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    req = urllib.request.Request(url, headers=dict(headers or {}, **({"Range": f"bytes={have}-"} if have else {})))
    op = opener or urllib.request.build_opener()
    try:
        r = op.open(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return dst
        raise
    if have and r.status != 206:
        have = 0  # server ignored Range: start over
    total = have + int(r.headers.get("Content-Length", 0))
    ctype = r.headers.get("Content-Type", "")
    if "text/html" in ctype:
        sys.exit(f"{url}: server returned HTML instead of a file (login failed or package not accessible)")
    with open(dst, "ab" if have else "wb") as f:
        done = have
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {os.path.basename(dst)}  {human(done)} / {human(total)}  {100 * done / max(total, 1):5.1f}%", end="", flush=True)
    print()
    return dst


def to_train_ids(src, dst):
    lab = np.array(Image.open(src))
    if lab.ndim == 3:
        lab = lab[..., 0]
    Image.fromarray(LUT[lab]).save(dst)


# ---------------------------------------------------------------- Cityscapes
def cityscapes(args):
    user, pw = os.environ.get("CITYSCAPES_USERNAME"), os.environ.get("CITYSCAPES_PASSWORD")
    if not (user and pw):
        sys.exit("set CITYSCAPES_USERNAME / CITYSCAPES_PASSWORD (register at https://www.cityscapes-dataset.com/register/)")
    import http.cookiejar
    import urllib.parse
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.open(CS_LOGIN)  # get session cookie
    data = urllib.parse.urlencode({"username": user, "password": pw, "submit": "Login"}).encode()
    op.open(urllib.request.Request(CS_LOGIN, data=data))
    if not any(c.name.startswith("PHPSESSID") or "cityscapes" in c.name.lower() for c in cj):
        print("warning: no session cookie after login; the download will tell")

    dl = os.path.join(args.out, "_download")
    for name, pid in CS_PACKAGES.items():
        print(f"[*] {name}")
        download(CS_FILE.format(pid), os.path.join(dl, name), opener=op)

    splits = ("val",) if not args.keep_train else ("train", "val", "test")
    for name in CS_PACKAGES:
        print(f"[*] extracting {name} ({', '.join(splits)})")
        with zipfile.ZipFile(os.path.join(dl, name)) as z:
            members = [m for m in z.namelist() if any(f"/{s}/" in m for s in splits) and m.endswith(".png")
                       and ("gtFine" not in m or m.endswith(("_labelIds.png", "_labelTrainIds.png")))]
            for i, m in enumerate(members):
                z.extract(m, args.out)
                if i % 200 == 0:
                    print(f"\r  {i}/{len(members)}", end="", flush=True)
            print()
    make_train_ids(args.out, splits)
    finish(args.out)


def make_train_ids(root, splits):
    n = 0
    for s in splits:
        for src in glob.glob(os.path.join(root, "gtFine", s, "*", "*_gtFine_labelIds.png")):
            dst = src.replace("_labelIds.png", "_labelTrainIds.png")
            if not os.path.exists(dst):
                to_train_ids(src, dst)
                n += 1
    print(f"[*] wrote {n} *_labelTrainIds.png")


# ---------------------------------------------------------------- KITTI
def kitti(args):
    dl = os.path.join(args.out, "_download", "data_semantics.zip")
    print("[*] KITTI semantic segmentation (313 MB)")
    download(KITTI_URL, dl)
    with zipfile.ZipFile(dl) as z:
        names = [m for m in z.namelist() if m.startswith("training/") and m.endswith(".png")
                 and ("/image_2/" in m or "/semantic/" in m)]
        for m in names:
            z.extract(m, os.path.join(args.out, "_raw"))
    img_dir = os.path.join(args.out, "leftImg8bit", "val", "kitti")
    lab_dir = os.path.join(args.out, "gtFine", "val", "kitti")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    n = 0
    for src in sorted(glob.glob(os.path.join(args.out, "_raw", "training", "image_2", "*.png"))):
        stem = os.path.splitext(os.path.basename(src))[0]
        lab = os.path.join(args.out, "_raw", "training", "semantic", stem + ".png")
        if not os.path.exists(lab):
            continue
        shutil.copy(src, os.path.join(img_dir, f"kitti_{stem}_leftImg8bit.png"))
        to_train_ids(lab, os.path.join(lab_dir, f"kitti_{stem}_gtFine_labelTrainIds.png"))
        n += 1
    print(f"[*] {n} image/label pairs (KITTI semantic uses the Cityscapes label ids; images are 1242x375)")
    finish(args.out)


# ---------------------------------------------------------------- ADE20K
def ade20k(args):
    """Scene-parsing benchmark val split (2000 images, 150 classes) + the official EfficientViT-B1 ADE20K checkpoint."""
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(os.path.dirname(here), "model", "evit_hw_golden", "assets", "efficientvit_seg_b1_ade20k.pt")
    if not os.path.exists(ckpt):
        print("[*] EfficientViT-B1 ADE20K checkpoint")
        download(ADE_CKPT_URL, ckpt, headers={"User-Agent": "Mozilla/5.0"})
    dl = os.path.join(args.out, "_download", "ADEChallengeData2016.zip")
    print("[*] ADEChallengeData2016.zip (~923 MB)")
    download(ADE_URL, dl)
    with zipfile.ZipFile(dl) as z:
        members = [m for m in z.namelist() if "/validation/" in m and m.endswith((".jpg", ".png"))]
        for i, m in enumerate(members):
            z.extract(m, args.out)
            if i % 500 == 0:
                print(f"\r  {i}/{len(members)}", end="", flush=True)
        print()
    root = os.path.join(args.out, "ADEChallengeData2016")
    imgs = glob.glob(os.path.join(root, "images", "validation", "*.jpg"))
    labs = glob.glob(os.path.join(root, "annotations", "validation", "*.png"))
    print(f"\n[done] {len(imgs)} val images, {len(labs)} annotations under {root}; checkpoint {ckpt}")
    print("next (fixed 512x512 input; the official number, 42.84 mIoU, uses an aspect-preserving resize, so expect a few points less for FP32 too):")
    print(f'  python host/eval_dataset.py --images "{root}/images/validation/*.jpg" --ade20k --out results/ade512 --backend model')
    print(f'  python host/export_batch.py --images "{root}/images/validation/*.jpg" --ade20k --out batch_ade512 --limit 30')


def finish(root):
    imgs = glob.glob(os.path.join(root, "leftImg8bit", "val", "*", "*_leftImg8bit.png"))
    labs = glob.glob(os.path.join(root, "gtFine", "val", "*", "*_gtFine_labelTrainIds.png"))
    print(f"\n[done] {len(imgs)} val images, {len(labs)} trainId labels under {root}")
    print("next:")
    print(f'  python host/eval_dataset.py --images "{root}/leftImg8bit/val/*/*_leftImg8bit.png" --cityscapes --out results/val --backend model')
    print(f'  python host/export_batch.py --images "{root}/leftImg8bit/val/*/*_leftImg8bit.png" --cityscapes --out batch_val --limit 50')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cityscapes", "kitti", "ade20k"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-train", action="store_true", help="cityscapes: also extract train/test splits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"cityscapes": cityscapes, "kitti": kitti, "ade20k": ade20k}[args.dataset](args)


if __name__ == "__main__":
    main()
