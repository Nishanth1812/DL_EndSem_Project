"""Build ./data for task_1-3 from the official TokaMark Hub release.

    python download.py                  # keep pulling batches for ~1 h, then stop
    python download.py --minutes 20     # shorter budget
    python download.py --batches 2      # only batch_00000.tar and batch_00001.tar

Why the Hub and not the Echo S3 bucket: one shot is ~285 separate S3 objects there, and that
endpoint serves ~20 objects/s whatever the worker count, so the 11.5k-shot set needs ~40 h.
The Hub ships one tar per ~200 shots, so the same data arrives at tens of MB/s.

Only what task_1-3 reads is kept: the 8 signals from tokamark's task_1-3.yaml, their time axes,
every 1-D array in the same groups (channel maps etc., 1 chunk each), and the group metadata.
That is ~65 objects per shot instead of ~285. float64 blosc chunks are rewritten as float32
zstd - the trainer casts to float32 anyway - which halves the bytes on disk.

Resumable: a finished shot carries a .done2 marker, processed batch tars are listed in
<cache>/done2.json, and data/shots_random.csv is rewritten after every batch. Delete data/<shot>.zarr
to redo one shot.
"""
import argparse
import io
import json
import os
import posixpath
import shutil
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import hf_hub_download
from tokamark.tools.path import RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE

RANDOM_SPLIT = RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
REPO = "UKAEA-IBM-STFC/tokamark-v1"
GROUPS = ("equilibrium", "magnetics", "pf_active", "summary")
SIGNALS = {  # task_1-3.yaml input/output names; everything 1-D in these groups is kept too
    "equilibrium": ("psi",),
    "magnetics": ("flux_loop_flux", "b_field_pol_probe_ccbv_field", "b_field_pol_probe_obr_field",
                  "b_field_pol_probe_obv_field", "b_field_tor_probe_saddle_voltage"),
    "pf_active": ("coil_current", "solenoid_current"),
    "summary": ("ip",),
}
DONE = ".done2"  # bumped when the on-disk layout changed, so older extractions are redone
NEW = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
OLD = numcodecs.Blosc()  # decode() reads the settings from each chunk's own header


def tars():
    """[(name, bytes)] of the batch archives on the Hub, in order."""
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=1", timeout=60) as r:
        tree = json.load(r)
    return [(t["path"], t["size"]) for t in tree if t["path"].startswith("batch_") and t["path"].endswith(".tar")]


def fetch(name, cache, log):
    """Download one batch tar into the Hub cache and return its local path.

    hf_hub_download with hf_xet installed streams the file in parallel chunks, resumes across
    drops and verifies it, which beats a single-connection GET on these multi-GB archives."""
    t = time.time()
    try:
        path = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache)
    except Exception as e:  # one bad batch should not end the run; the next one may be fine
        log(f"    download failed ({type(e).__name__}: {e})")
        return None
    size = os.path.getsize(path)
    log(f"    {size / 1e9:.2f} GB in {(time.time() - t) / 60:.1f} min "
        f"({size / 1e6 / max(1e-9, time.time() - t):.0f} MB/s)")
    return path


def members(zf):
    """{path inside the store: name in the zip}; the Hub zips wrap everything in '<shot>.zarr/'."""
    out = {}
    for n in zf.namelist():
        if n.endswith("/"):
            continue
        head, sep, rest = n.partition("/")
        out[rest if sep and head.endswith(".zarr") else n] = n
    return out


def plan(zf, m):
    """(members to keep, {array dir: narrowed .zarray}, consolidated metadata or None)."""
    names = set(m)
    read = lambda n: zf.read(m[n])
    keep, narrow, meta = set(), {}, None
    if ".zmetadata" in names:
        meta = json.loads(read(".zmetadata"))
    for n in names:
        parts = n.split("/")
        if len(parts) == 1 and n in (".zgroup", ".zattrs", ".zmetadata"):
            keep.add(n)
        elif len(parts) == 2 and parts[0] in GROUPS and parts[1] in (".zgroup", ".zattrs"):
            keep.add(n)
        elif len(parts) >= 3 and parts[0] in GROUPS and parts[1] in SIGNALS[parts[0]]:
            keep.add(n)  # a task_1-3 signal: every chunk plus its .zarray/.zattrs
    # 1-D arrays (time axes, channel maps) are one or two chunks each, and the loader may need them
    for arr in {n.rsplit("/", 1)[0] for n in names if n.count("/") == 2 and n.endswith("/.zarray")}:
        g = arr.split("/")[0]
        if g in GROUPS and f"{arr}/.zarray" not in keep and len(json.loads(read(f"{arr}/.zarray"))["shape"]) == 1:
            keep.update(m2 for m2 in names if m2.startswith(f"{arr}/"))
    for arr in {n.rsplit("/", 1)[0] for n in keep if n.count("/") == 2 and n.endswith("/.zarray")}:
        za = json.loads(read(f"{arr}/.zarray"))
        if za["dtype"] == "<f8" and not za.get("filters") and (za["compressor"] or {}).get("id") == "blosc":
            za.update(dtype="<f4", compressor=NEW.get_config())
            narrow[arr] = za
    return keep, narrow, meta


def write_shot(zf, dst):
    """Materialise one shot's zip into dst, narrowing float64 chunks as it goes."""
    m = members(zf)
    keep, narrow, meta = plan(zf, m)
    # some Hub zips (e.g. shot 17097) hold a group's arrays but no <group>/.zgroup, so the group cannot be opened
    groups = {n.split("/")[0] for n in keep if n.count("/") >= 2 and n.split("/")[0] in GROUPS}
    for g in groups:
        os.makedirs(os.path.join(dst, g), exist_ok=True)
        if f"{g}/.zgroup" not in keep:
            open(os.path.join(dst, g, ".zgroup"), "w").write('{"zarr_format": 2}')
    for n in keep:
        out = os.path.join(dst, *n.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        data = zf.read(m[n])
        arr = n.rsplit("/", 1)[0]
        if arr in narrow and not n.endswith(("/.zarray", "/.zattrs", "/.zgroup")):
            data = NEW.encode(np.frombuffer(OLD.decode(data), dtype="<f8").astype("<f4"))
        elif arr in narrow:  # .zarray now describes float32, not float64
            data = json.dumps(narrow[arr], indent=4).encode()
        open(out, "wb").write(data)
    if meta is not None:  # consolidated metadata must list exactly what is on disk, or zarr raises on lookup
        meta["metadata"] = {k: v for k, v in meta["metadata"].items() if k in keep}
        for g in groups:
            meta["metadata"].setdefault(f"{g}/.zgroup", {"zarr_format": 2})
        for arr, za in narrow.items():
            meta["metadata"][f"{arr}/.zarray"] = za
        json.dump(meta, open(os.path.join(dst, ".zmetadata"), "w"), indent=4)


def check(dst):
    """Raise unless the loader will find every task_1-3 signal in this shot (same open call as MAST_dataset).
    A shot that fails here is left out of the split files instead of crashing training hours in."""
    for g, names in SIGNALS.items():
        with xr.open_zarr(dst, group=g) as ds:
            missing = [v for v in names if v not in ds]
            if missing:
                raise KeyError(f"{g} lacks {missing}")


def write_splits(out, only=None):
    """Write the official random split restricted to shots on disk (and to `only`, if given)."""
    have = {int(d.split(".")[0]) for d in os.listdir(out)
            if d.endswith(".zarr") and os.path.exists(os.path.join(out, d, DONE))}
    if only is not None:
        have &= only
    table = pd.read_csv(RANDOM_SPLIT)
    table[table.shot_id.isin(have)].to_csv(os.path.join(out, "shots_random.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=None, help="max batches this run (default: no limit)")
    ap.add_argument("--minutes", type=float, default=60, help="wall-clock budget for downloading")
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default=".cache", help="where batch tars are staged")
    ap.add_argument("--max-gb", type=float, default=6, help="skip batches bigger than this (a few are 15-37 GB)")
    ap.add_argument("--workers", type=int, default=8, help="parallel shot extractions")
    ap.add_argument("--keep-tars", action="store_true", help="keep the staged tars instead of deleting them")
    ap.add_argument("--shots-from", help="dir holding a previous run's shots_random.csv: fetch exactly those shots, "
                                         "so every model is tested on the same clean test set")
    a = ap.parse_args()
    only = None
    if a.shots_from:
        only = set(pd.read_csv(f"{a.shots_from}/shots_random.csv").shot_id.astype(int))

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    deadline = t0 + 60 * a.minutes
    batch_list = tars()
    done_file = os.path.join(a.cache, "done2.json")
    done = json.load(open(done_file)) if os.path.exists(done_file) else []
    todo = [(n, s) for n, s in batch_list if n not in done and s <= a.max_gb * 1e9]
    skipped = [(n, s) for n, s in batch_list if n not in done and s > a.max_gb * 1e9]
    print(f"{len(batch_list)} batches on the Hub, {len(done)} already processed, "
          f"{len(todo)} to go; budget {a.minutes:.0f} min"
          + (f"; skipping {len(skipped)} batches over {a.max_gb:.0f} GB "
             f"({sum(s for _, s in skipped) / 1e9:.0f} GB)" if skipped else ""), flush=True)

    log = lambda m: print(m, flush=True)
    for i, (name, size) in enumerate(todo, 1):
        if a.batches is not None and i > a.batches:
            break
        if time.time() > deadline:
            log(f"budget spent after {len(done)} batches; stopping")
            break
        log(f"[{i}/{len(todo)}] {name} {size / 1e9:.2f} GB")
        path = fetch(name, a.cache, log)
        if path is None:
            continue

        t1 = time.time()
        with tarfile.open(path, "r:") as tf:  # index only; the workers read their own byte range
            todo_shots = [m for m in tf if m.isfile() and m.name.endswith(".zarr.zip")]
        n_ok = n_skip = n_bad = 0

        def one(m):
            shot = posixpath.basename(m.name).split(".zarr.zip")[0]
            dst = os.path.join(a.out, f"{shot}.zarr")
            if only is not None and int(shot) not in only:
                return "skip", shot
            if os.path.exists(os.path.join(dst, DONE)):
                return "skip", shot
            tmp = f"{dst}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                with open(path, "rb") as f:  # one seek per shot, so threads never share a tar handle
                    f.seek(m.offset_data)
                    with zipfile.ZipFile(io.BytesIO(f.read(m.size))) as zf:
                        write_shot(zf, tmp)
                check(tmp)
                open(os.path.join(tmp, DONE), "w").close()
                shutil.rmtree(dst, ignore_errors=True)
                os.replace(tmp, dst)
                return "ok", shot
            except Exception as e:  # a broken shot must not kill the batch
                shutil.rmtree(tmp, ignore_errors=True)
                return f"FAILED {shot}: {e}", shot

        with ThreadPoolExecutor(a.workers) as ex:  # blosc + zlib release the GIL, so threads scale
            for r, shot in ex.map(one, todo_shots):
                if r == "ok":
                    n_ok += 1
                elif r == "skip":
                    n_skip += 1
                else:
                    n_bad += 1
                    log(f"    {r}")
                if (n_ok + n_skip) % 50 == 0:
                    log(f"    {n_ok} kept, {n_skip} already had, {n_bad} failed | "
                        f"{(n_ok + n_skip) / max(1e-9, time.time() - t1):.1f} shots/s")
        done.append(name)
        json.dump(done, open(done_file, "w"))
        if not a.keep_tars:  # the Hub cache holds a copy of every tar we have pulled so far
            shutil.rmtree(os.path.join(a.cache, "datasets--UKAEA-IBM-STFC--tokamark-v1"), ignore_errors=True)
        log(f"    batch done: {n_ok} kept, {n_skip} skipped, {n_bad} failed, "
            f"{(time.time() - t1) / 60:.1f} min; {(time.time() - t0) / 60:.0f} min total")
        write_splits(a.out, only)  # after every batch, so a kill still leaves runnable split files

    write_splits(a.out, only)
    table = pd.read_csv(os.path.join(a.out, "shots_random.csv"))
    n = {c: int(table[c].sum()) for c in ("train", "val", "test")}
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(a.out) for f in fs) / 1e9
    print(f"\n{len(table)} shots on disk ({total:.1f} GB): train {n['train']}, val {n['val']}, test {n['test']} "
          f"(random split)\nwrote data/shots_random.csv - run.py uses it as the split file", flush=True)


if __name__ == "__main__":
    main()
