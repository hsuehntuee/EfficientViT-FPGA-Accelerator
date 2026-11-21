import argparse
import os
import time
import struct
import mmap
import numpy as np

XDMA_USER_PATH = "/dev/xdma0_user"
XDMA_H2C_PATH  = "/dev/xdma0_h2c_0"
XDMA_C2H_PATH  = "/dev/xdma0_c2h_0"

# Must match the address editor of design_1.bd (see scripts/vivado/update_bd.tcl).
BRAM_MAP = {
    "BRAM_B":       (0xC0000000, 64 * 1024),
    "BRAM_DW_B":    (0xC2000000, 64 * 1024),
    "BRAM_DW_W":    (0xC4000000, 1024 * 1024),   # MBConv DW at word 0, MS DW at word 16384 (byte 524288)
    "BRAM_PW_B":    (0xC6000000, 64 * 1024),
    "BRAM_PW_W":    (0xC8000000, 512 * 1024),
    "BRAM_W":       (0xCA000000, 512 * 1024),
    "BRAM_X":       (0xCC000000, 128 * 1024),
    "BRAM_Y":       (0xCE000000, 128 * 1024),
    "BRAM_MLA_B":   (0xD0000000, 32 * 1024),
    "BRAM_MLA_W":   (0xD2000000, 1024 * 1024),
    "BRAM_MS_PW_B": (0xD4000000, 8 * 1024),
    "BRAM_MS_PW_W": (0xD6000000, 32 * 1024),
}

# hw_export file suffix -> BRAM
WEIGHT_FILES = [
    ("x", "BRAM_X"), ("w", "BRAM_W"), ("b", "BRAM_B"),
    ("dw_w", "BRAM_DW_W"), ("dw_b", "BRAM_DW_B"),
    ("pw_w", "BRAM_PW_W"), ("pw_b", "BRAM_PW_B"),
    ("mla_w", "BRAM_MLA_W"), ("mla_b", "BRAM_MLA_B"),
    ("ms_pw_w", "BRAM_MS_PW_W"), ("ms_pw_b", "BRAM_MS_PW_B"),
]

REG_START, REG_DONE, REG_CYCLES = 0x00, 0x04, 0x08

def load_binary_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    with open(filepath, 'rb') as f:
        return f.read()

def write_bram(fd_h2c, name, data_bytes):
    offset, max_size = BRAM_MAP[name]
    if len(data_bytes) > max_size:
        raise ValueError(f"{name}: {len(data_bytes)} bytes does not fit in {max_size}")
    bytes_w = 0
    chunk_size = 65536
    while bytes_w < len(data_bytes):
        write_size = min(chunk_size, len(data_bytes) - bytes_w)
        ret = os.pwrite(fd_h2c, data_bytes[bytes_w:bytes_w+write_size], offset + bytes_w)
        if ret <= 0: raise OSError(f"XDMA write aborted at {name}")
        bytes_w += ret

def read_bram(fd_c2h, name, req_bytes):
    offset, max_size = BRAM_MAP[name]
    actual_size = min(req_bytes, max_size)
    result = bytearray()
    bytes_r = 0
    chunk_size = 65536
    while bytes_r < actual_size:
        read_size = min(chunk_size, actual_size - bytes_r)
        chunk = os.pread(fd_c2h, read_size, offset + bytes_r)
        if not chunk: break
        result.extend(chunk)
        bytes_r += len(chunk)
    return bytes(result)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hw_export_512")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for done")
    ap.add_argument("--save", default=None, help="write the raw BRAM_Y contents to this file")
    ap.add_argument("--settle", type=float, default=0.0, help="seconds to wait between the last BRAM write and start")
    ap.add_argument("--write-twice", action="store_true", help="write every BRAM twice before starting")
    ap.add_argument("--verify-bram", action="store_true", help="read every BRAM back over C2H and compare before starting")
    args = ap.parse_args()

    try:
        fd_user = os.open(XDMA_USER_PATH, os.O_RDWR | os.O_SYNC)
        user_map = mmap.mmap(fd_user, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        fd_h2c  = os.open(XDMA_H2C_PATH, os.O_WRONLY)
        fd_c2h  = os.open(XDMA_C2H_PATH, os.O_RDONLY)
    except Exception as e:
        print(f"[FATAL] PCIe XDMA Driver Error: {e}")
        return
    def write_reg(offset, val):
        user_map.seek(offset)
        user_map.write(struct.pack("<I", val))

    def read_reg(offset):
        user_map.seek(offset)
        return struct.unpack("<I", user_map.read(4))[0]

    print(f"[*] Loading block {args.block} from '{args.dir}/'...")
    path = lambda n: os.path.join(args.dir, f"block_{args.block}_{n}.bin")
    blobs = [(bram, load_binary_file(path(n))) for n, bram in WEIGHT_FILES]
    bytes_y_g = load_binary_file(path("y_golden"))

    print("[*] Writing BRAMs...")
    for _ in range(2 if args.write_twice else 1):
        for bram, data in blobs:
            write_bram(fd_h2c, bram, data)
    if args.verify_bram:
        bad = 0
        for bram, data in blobs:
            back = read_bram(fd_c2h, bram, len(data))
            if back != data:
                nd = sum(1 for a, b in zip(back, data) if a != b)
                first = next(i for i, (a, b) in enumerate(zip(back, data)) if a != b)
                print(f"[VERIFY] {bram}: {nd} of {len(data)} bytes differ, first at byte {first} (word {first // 32})")
                bad += 1
        print(f"[VERIFY] {'all BRAMs read back identical' if not bad else str(bad) + ' BRAM(s) differ'}")
    if args.settle > 0:
        time.sleep(args.settle)

    print("[*] Start...")
    write_reg(REG_START, 0)
    write_reg(REG_START, 1)
    t0 = time.time()
    while not (read_reg(REG_DONE) & 1):
        if time.time() - t0 > args.timeout:
            print(f"[FATAL] done never asserted within {args.timeout} s (cycle count so far: {read_reg(REG_CYCLES)})")
            write_reg(REG_START, 0)
            return
        time.sleep(0.001)
    cycles = read_reg(REG_CYCLES)
    print(f"[+] Done. Clock cycles: {cycles}")

    y_harvested = read_bram(fd_c2h, "BRAM_Y", len(bytes_y_g))
    write_reg(REG_START, 0)
    if args.save:
        with open(args.save, "wb") as f:
            f.write(y_harvested)

    y_hw = np.frombuffer(y_harvested, dtype=np.int16).astype(np.int32)
    y_g = np.frombuffer(bytes_y_g, dtype=np.int16).astype(np.int32)

    # y_golden is the float model's output quantised, so it is never bit-exact.
    # Compare the numbers below with `python model/rtl_model.py` ("RTL fixed" row).
    d = np.abs(y_hw - y_g)
    print("\n================================================================")
    print(f" Golden   (first 10): {y_g[:10]}")
    print(f" Hardware (first 10): {y_hw[:10]}")
    print(f" mean |HW - golden| = {d.mean():.2f} LSB   max = {d.max()}   "
          f"rel = {100 * d.mean() / max(np.abs(y_g).mean(), 1e-9):.2f}%")
    print("================================================================\n")

    user_map.close(); os.close(fd_user); os.close(fd_h2c); os.close(fd_c2h)

if __name__ == "__main__":
    main()
