#!/usr/bin/env python
"""TokaMark task_1-3 CNN baseline on Kaggle: prep | train | eval | plot | summary.

Usage: python pipeline.py {prep,train,eval,plot,summary} [--split random|temporal]
All settings come from environment variables (see CFG below), set by the notebook.

Protocol notes (deviations from the official run_training.py / run_evaluation.py):
- Custom step-based training loop mirroring src/trainer.py:BatchStepTrainer (same
  Adam(lr, wd=1e-4) + MultiOutputMSELoss, validate_every, patience, drop_last=True),
  extended with: wall-clock deadline, periodic + atomic checkpointing, resume from
  last.pt, CUDA-OOM reporting, and non-finite-loss skipping.
- Shots are a random subset OF THE OFFICIAL SPLIT LISTS (membership never mixed),
  chosen under a byte budget; official stats/outlier/split files always match the
  split protocol. test uses shuffle_windows=False (official eval behaviour).
"""
import os, sys, json, time, math, random, shutil, zipfile, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

T0 = float(os.environ.get("TM_T0", time.time()))  # session start (set by notebook)


def env(k, d):
    return type(d)(os.environ.get(k, d))


CFG = dict(
    task=env("TM_TASK", "task_1-3"),
    seed=env("TM_SEED", 23),
    repo_id="UKAEA-IBM-STFC/tokamark-v1",
    work=env("TM_WORK", "/kaggle/working"),
    data_root=env("TM_DATA_ROOT", "/tmp/tokamark_data"),
    tokamark_dir=env("TM_TOKAMARK", "/kaggle/working/tokamark"),
    baseline_dir=env("TM_BASELINE", "/kaggle/working/tokamark_baseline"),
    data_budget_gb=env("TM_DATA_GB", 40.0),
    prep_deadline_h=env("TM_PREP_DEADLINE_H", 1.75),      # hours since T0: stop downloading
    train_deadline_h=env("TM_TRAIN_DEADLINE_H", 8.0),     # hours since T0: stop training
    batch_size=env("TM_BATCH", 512),
    validate_every=env("TM_VALIDATE_EVERY", 100),
    patience=env("TM_PATIENCE", 10),
    lr=1e-4,
    weight_decay=1e-4,                                    # official Adam weight decay
    max_steps=env("TM_MAX_STEPS", 10_000_000),
    dl_workers=env("TM_DL_WORKERS", -1),                  # -1 -> auto; 0 -> main process only; N -> N workers
    use_dp=env("TM_USE_DP", "auto"),                      # auto|0|1
    ckpt_every_s=env("TM_CKPT_EVERY_S", 600),
    force_reselect=env("TM_FORCE_RESELECT", "0") == "1",
    plot_windows=env("TM_PLOT_WINDOWS", 3),
    git_revs=env("TM_GIT_REVS", "unknown"),
    index_threads=env("TM_INDEX_THREADS", 24),
    index_stop_gb=env("TM_INDEX_STOP_GB", 0.0),           # 0 -> auto: max(3, 1.6 x data budget)
    index_min_tars=env("TM_INDEX_MIN_TARS", 6),           # spread subset across >= this many archives
)


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f}m]", *a, flush=True)


def paths(split):
    out = os.path.join(CFG["work"], "out", split)
    return dict(
        out=out,
        data=os.path.join(CFG["data_root"], split),
        index=os.path.join(CFG["work"], "out", "tar_index.json"),
        sel=os.path.join(out, "selection.json"),
        model_dir=os.path.join(out, "model") + "/",
        results=os.path.join(out, "results") + "/",
        figures=os.path.join(out, "figures") + "/",
    )


def setup_imports():
    for p in (os.path.join(CFG["tokamark_dir"], "src"), CFG["baseline_dir"]):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(CFG["baseline_dir"])


def split_files(split):
    """Official split CSV + stats + outlier file; ALWAYS matched per protocol."""
    setup_imports()
    from tokamark.tools.path import (RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE as RS, RANDOM_SPLIT_SIGNALS_STATS_FILE as RT,
                                     TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE as TS, TEMPORAL_SPLIT_SIGNALS_STATS_FILE as TT)
    from MAST_tools.utils.path_utils import RANDOM_SPLIT_OUTLIER_METADATA_FILE as RO, TEMPORAL_SPLIT_OUTLIER_METADATA_FILE as TO
    if split == "random":
        return RS, RT, RO
    if split == "temporal":
        return TS, TT, TO
    raise ValueError(f"unknown split {split}")


def patch_stats_module(stats):
    """Point get_signals_metadata at the stats file of the CURRENT split protocol.

    Upstream get_task_metadata() always reads the RANDOM-split stats file, and
    tokamark.evaluator binds get_signals_metadata at import time; patch both module
    references so metadata and NRMSE normalisation always match the protocol.
    """
    setup_imports()
    import tokamark.tasks as _tt
    import tokamark.evaluator as _te
    if getattr(_tt, "_tm_patched", None) == stats:
        return
    orig = getattr(_tt, "_tm_orig", None) or _tt.get_signals_metadata
    _tt._tm_orig = orig
    patched = lambda file_path=stats, _o=orig: _o(file_path=file_path)  # noqa: E731
    _tt.get_signals_metadata = patched
    _te.get_signals_metadata = patched
    _tt._tm_patched = stats
    log(f"patched get_signals_metadata -> {os.path.basename(stats)}")


def first_batch(loader, tries=100):
    """Return the first non-empty batch (collate_fn may yield None for empty batches)."""
    it = iter(loader)
    for _ in range(tries):
        b = next(it, None)
        if b is not None:
            return b
    raise RuntimeError("loader produced no non-empty batch")


def first_batch_timed(loader, tag="first batch", tries=100, timeout_s=1800):
    """first_batch with a heartbeat + hard wall-clock cap.

    A silent hang here (workers stuck on IPC / fork-after-CUDA / slow zarr reads) used
    to look like training forever; now it is visible every 30 s and fails loudly.
    """
    import threading
    res, done = {}, threading.Event()

    def _work():
        try:
            res["b"] = first_batch(loader, tries)
        except BaseException as e:  # noqa
            res["e"] = e
        finally:
            done.set()

    threading.Thread(target=_work, daemon=True).start()
    t0 = time.time()
    while not done.wait(30):
        log(f"  waiting for {tag}... {time.time() - t0:.0f}s (workers are still producing it)")
        if time.time() - t0 > timeout_s:
            raise RuntimeError(
                f"{tag} did not arrive within {timeout_s}s. This is a dataloader hang, not training. "
                f"Try TM_DL_WORKERS=0 (main-process loading, no workers/IPC) or a smaller TM_BATCH "
                f"({CFG['batch_size']} -> 128/64).")
    if "e" in res:
        raise res["e"]
    return res["b"]


# ======================================================================================
# PREP: index tars with range requests, select shots per official split, fetch members
# ======================================================================================
def hf_headers():
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


import threading
_TLS = threading.local()


def _session():
    """Per-thread requests.Session: TLS keep-alive turns ~1-3 s/request into ~0.2 s."""
    s = getattr(_TLS, "session", None)
    if s is None:
        import requests
        s = _TLS.session = requests.Session()
    return s


class _Deadline(Exception):
    pass


def ranged_get(url, a, b, tries=6, deadline=None):
    last = None
    for i in range(tries):
        if deadline is not None and time.time() > deadline:
            raise _Deadline()
        try:
            r = _session().get(url, headers={**hf_headers(), "Range": f"bytes={a}-{b}"}, timeout=60)
            if r.status_code in (200, 206):
                if r.status_code == 200 and len(r.content) > (b - a + 1):  # server ignored Range
                    raise RuntimeError("Range ignored by server")
                return r.content
            last = f"HTTP {r.status_code}"
        except _Deadline:
            raise
        except Exception as e:  # noqa
            last = repr(e)
        time.sleep(min(30, 2 ** i))
    raise RuntimeError(f"range {a}-{b} failed: {last}")


def make_ranged_reader(url, total, deadline=None, span=4096):
    """Small-read-optimised reader for parse_tar_headers.

    Header walks issue many 512-byte reads a few KB apart; HF resolve latency makes
    one-request-per-read prohibitively slow, so each read pulls a `span`-byte window
    and serves nearby reads from a one-window cache.
    """
    buf = {"start": -1, "data": b""}

    def read(a, n):
        end = min(a + n, total)
        if a >= total or end <= a:
            return b""
        if buf["start"] <= a and end <= buf["start"] + len(buf["data"]):
            o = a - buf["start"]
            return buf["data"][o:o + (end - a)]
        b = min(a + max(span, n) - 1, total - 1)
        data = ranged_get(url, a, b, deadline=deadline)
        buf["start"], buf["data"] = a, data
        return data[: end - a]

    return read


def _octal_or_b256(field):
    if field[0] & 0x80:
        return int.from_bytes(field, "big") & ((1 << (8 * len(field) - 1)) - 1)
    s = field.split(b"\0")[0].strip()
    return int(s, 8) if s else 0


def parse_tar_headers(read, total):
    """Walk a tar using read(offset, length)->bytes. Returns [(name, data_offset, size)]."""
    members, off, pending = [], 0, None
    while off + 512 <= total:
        h = read(off, 512)
        if len(h) < 512 or h.count(0) == 512:
            break
        name = h[:100].split(b"\0")[0].decode("utf8", "replace")
        if h[257:262] == b"ustar" and h[345:500].strip(b"\0"):
            name = h[345:500].split(b"\0")[0].decode("utf8", "replace") + "/" + name
        size = _octal_or_b256(h[124:136])
        flag = h[156:157]
        data_off = off + 512
        nxt = data_off + ((size + 511) // 512) * 512
        if flag in (b"L", b"x", b"g", b"K"):
            data = read(data_off, size)
            if flag == b"L":
                pending = data.split(b"\0")[0].decode("utf8", "replace")
            elif flag == b"x":
                for rec in data.split(b"\n"):
                    if b" path=" in rec:
                        pending = rec.split(b" path=", 1)[1].decode("utf8", "replace")
            off = nxt
            continue
        if flag in (b"0", b"\0"):
            members.append((pending or name, data_off, size))
        pending = None
        off = nxt
    return members


def build_index(deadline_ts):
    """Index HF tars via range-request header walks. Resumable (index saved per tar).

    Tars are walked in a seeded-shuffled order and indexing stops early once the
    indexed shot bytes cover the download budget (x1.6 margin) - we only need enough
    archives to draw the subset, not the whole 500+ GB repo. Hard-stops at deadline_ts;
    a re-run continues from the saved index.
    """
    from huggingface_hub import HfApi, hf_hub_url
    P = paths("random")
    os.makedirs(os.path.dirname(P["index"]), exist_ok=True)
    idx = json.load(open(P["index"])) if os.path.exists(P["index"]) else {"tars": {}, "shots": {}}
    api = HfApi()
    files = [f for f in api.list_repo_tree(CFG["repo_id"], repo_type="dataset", recursive=True)
             if getattr(f, "size", None) is not None and f.path.endswith(".tar")]
    files.sort(key=lambda f: f.path)
    log(f"HF repo lists {len(files)} tar files, {sum(f.size for f in files) / 1e9:.1f} GB total")
    indexed_bytes = sum(v[2] for v in idx["shots"].values())
    stop_gb = CFG["index_stop_gb"] or max(3.0, 1.6 * CFG["data_budget_gb"])
    min_tars = max(1, CFG["index_min_tars"])
    log(f"index: {len(idx['shots'])} shots / {indexed_bytes / 1e9:.1f} GB indexed (cached) across {len(idx['tars'])} tars; "
        f"stop submitting beyond {stop_gb:.1f} GB and >= {min_tars} tars")
    todo = [f for f in files if f.path not in idx["tars"]]
    random.Random(CFG["seed"]).shuffle(todo)  # seeded spread across the archive

    def stop_reached():
        return indexed_bytes >= stop_gb * 1e9 and len(idx["tars"]) >= min_tars

    def work(f):
        url = hf_hub_url(CFG["repo_id"], f.path, repo_type="dataset")
        read = make_ranged_reader(url, f.size, deadline=deadline_ts)
        nreq = [0]
        mem = parse_tar_headers(lambda a, n: (nreq.__setitem__(0, nreq[0] + 1), read(a, n))[1], f.size)
        return f.path, url, mem

    with ThreadPoolExecutor(CFG["index_threads"]) as ex:
        it = iter(todo)
        pending = {}

        def submit():
            if stop_reached():
                return False
            try:
                f = next(it)
            except StopIteration:
                return False
            fu = ex.submit(work, f)
            pending[fu] = f.path
            return True

        need = stop_gb * 1e9 - indexed_bytes
        cum, n_need = 0, 0
        if need > 0:
            for f in todo:
                cum += f.size
                n_need += 1
                if cum >= need:
                    break
        min_tars_left = max(0, min_tars - len(idx["tars"]))
        initial = min(CFG["index_threads"], max(n_need, min_tars_left), len(todo))
        log(f"submitting {initial} tars now (need ~{n_need} for budget, {min_tars_left} for archive spread)")
        for _ in range(initial):
            if not submit():
                break
        last_tick = time.time()
        while pending:
            if time.time() > deadline_ts:
                log("index deadline reached: cancelling remaining tar walks")
                for fu in pending:
                    fu.cancel()
                break
            if stop_reached():
                log(f"enough archives indexed ({indexed_bytes / 1e9:.1f} GB across {len(idx['tars'])} tars): "
                    "cancelling remaining tar walks")
                for fu in pending:
                    fu.cancel()
                break
            try:
                done_fu = next(as_completed(list(pending), timeout=30), None)
            except TimeoutError:
                if time.time() - last_tick > 120:
                    log(f"  indexing... {len(pending)} tars in flight, {len(idx['tars'])} done, {indexed_bytes / 1e9:.1f} GB")
                    last_tick = time.time()
                continue
            fpath = pending.pop(done_fu)
            try:
                path, url, mem = done_fu.result()
            except _Deadline:
                log(f"  {fpath}: walk aborted at indexing deadline (will resume next run)")
                continue
            except Exception as e:  # noqa
                log(f"  index FAILED for {fpath}: {repr(e)[:200]}")
                submit()
                continue
            n = 0
            for name, off, size in mem:
                base = os.path.basename(name)
                if not base.endswith(".zarr.zip"):
                    continue
                try:
                    sid = int(base[: -len(".zarr.zip")])
                except ValueError:
                    log("  skip unparsable member", name)
                    continue
                if str(sid) not in idx["shots"]:
                    idx["shots"][str(sid)] = [url, off, size]
                    indexed_bytes += size
                n += 1
            idx["tars"][path] = n
            log(f"  indexed {path}: {n} shots ({indexed_bytes / 1e9:.1f} GB total indexed)")
            json.dump(idx, open(P["index"], "w"))
            submit()
    missing = [f.path for f in files if f.path not in idx["tars"]]
    if missing and not stop_reached():
        log(f"WARNING: {len(missing)} tars not indexed (deadline/network?): {missing[:5]} - re-run prep to continue")
    log(f"index: {len(idx['shots'])} shots across {len(idx['tars'])} tars ({indexed_bytes / 1e9:.1f} GB)")
    return idx


def _quota_select(budget, official, avail, rng):
    """Random-priority selection of shots under a byte quota (order stays random)."""
    rng.shuffle(avail)
    used, chosen = 0, []
    for s in avail:
        sz = s[1]
        if used + sz > budget:
            continue  # keep scanning: smaller shots may still fit
        chosen.append(s[0])
        used += sz
    return chosen, used


def choose_shots(split, idx):
    from tokamark.data_split import read_data_split_csv
    csv, _, _ = split_files(split)
    tr, te, va = read_data_split_csv(csv)
    official = {"train": [int(s) for s in tr], "val": [int(s) for s in va], "test": [int(s) for s in te]}
    total = sum(len(v) for v in official.values())
    free = shutil.disk_usage(paths(split)["data"]).free
    budget = min(CFG["data_budget_gb"] * 1e9, max(0.0, free - 6e9))
    log(f"[{split}] official counts: " + ", ".join(f"{k}={len(v)}" for k, v in official.items()) + f" (total {total}); byte budget {budget / 1e9:.1f} GB")
    rng = random.Random(CFG["seed"])
    MIN_SHOTS = 5  # matches the fatal guard in cmd_prep
    sel, report = {}, {}
    for k, shots in official.items():
        avail = [(s, idx["shots"][str(s)][2]) for s in shots if str(s) in idx["shots"]]
        quota = budget * len(shots) / total
        rng.shuffle(avail)
        # guarantee >= MIN_SHOTS per split even when the byte quota is tiny: the 5
        # smallest available (cost-bounded), then fill the rest of the quota randomly.
        guard = sorted(avail, key=lambda t: t[1])[:MIN_SHOTS]
        guard_ids = {s for s, _ in guard}
        used1 = sum(sz for _, sz in guard)
        rest = [t for t in avail if t[0] not in guard_ids]
        extra, used2 = _quota_select(max(0.0, quota - used1), shots, rest, rng)
        chosen = [s for s, _ in guard] + extra
        used = used1 + used2
        sel[k] = chosen
        report[k] = dict(official=len(shots), in_archives=len(avail), selected=len(chosen), gb=round(used / 1e9, 2),
                         fraction_of_official=round(len(chosen) / max(1, len(shots)), 4))
    log(f"[{split}] selection: {json.dumps(report)}")
    return sel, report


def fetch_member(sid, meta, data_dir):
    url, off, size = meta
    final = os.path.join(data_dir, f"{sid}.zarr")
    if os.path.isdir(final):
        return sid, True, "cached"
    zpath = os.path.join(data_dir, f"{sid}.zip.part")
    tmpd = os.path.join(data_dir, f"{sid}.extract")
    try:
        with open(zpath, "wb") as fh:
            pos, CH = off, 64 * 2 ** 20
            while pos < off + size:
                end = min(pos + CH, off + size) - 1
                fh.write(ranged_get(url, pos, end))
                pos = end + 1
        if os.path.getsize(zpath) != size:
            raise RuntimeError("size mismatch")
        shutil.rmtree(tmpd, ignore_errors=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpd)
        root = tmpd
        if not (os.path.exists(os.path.join(root, "zarr.json")) or os.path.exists(os.path.join(root, ".zgroup"))):
            kids = [d for d in os.listdir(tmpd) if os.path.isdir(os.path.join(tmpd, d))]
            if len(kids) == 1 and (os.path.exists(os.path.join(tmpd, kids[0], "zarr.json")) or os.path.exists(os.path.join(tmpd, kids[0], ".zgroup"))):
                root = os.path.join(tmpd, kids[0])
            else:
                raise RuntimeError("no zarr root found in archive")
        import zarr
        zarr.open_group(root, mode="r")  # must open
        os.replace(root, final)
        return sid, True, "ok"
    except Exception as e:  # noqa
        return sid, False, repr(e)[:200]
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
        if os.path.exists(zpath):
            os.remove(zpath)


def cmd_prep(split):
    setup_imports()
    P = paths(split)
    os.makedirs(P["data"], exist_ok=True)
    os.makedirs(P["out"], exist_ok=True)
    deadline = T0 + CFG["prep_deadline_h"] * 3600
    idx = build_index(deadline)

    reuse = None
    if os.path.exists(P["sel"]) and not CFG["force_reselect"]:
        prev = json.load(open(P["sel"]))
        if prev.get("split") == split and min((len(v) for v in prev.get("shots", {}).values()), default=0) >= 5:
            reuse = prev
            log("reusing previous selection.json "
                f"({json.dumps({k: len(v) for k, v in reuse['shots'].items()})}); "
                "set TM_FORCE_RESELECT=1 to re-select shots")
    if reuse:
        sel, report = reuse["shots"], reuse["report"]
        for k in sel:
            report.setdefault(k, {})
            report[k]["official"] = report[k].get("official", len(sel[k]))
    else:
        sel, report = choose_shots(split, idx)

    # round-robin across splits so a deadline cut keeps the proportions; skip cached shots
    order, lists = [], {k: [s for s in v if not os.path.isdir(os.path.join(P["data"], f"{s}.zarr"))] for k, v in sel.items()}
    n_cached = sum(len(v) for v in sel.values()) - sum(len(v) for v in lists.values())
    if n_cached:
        log(f"{n_cached} selected shots already on disk (cached)")
    while any(lists.values()):
        for k in ("train", "val", "test"):
            if lists[k]:
                order.append((k, lists[k].pop()))
    done, failed = set(), {}
    nworkers = 6
    with ThreadPoolExecutor(nworkers) as ex:
        it = iter(order)
        futs = {}
        def submit():
            try:
                k, s = next(it)
            except StopIteration:
                return False
            futs[ex.submit(fetch_member, s, idx["shots"][str(s)], P["data"])] = (k, s)
            return True
        for _ in range(nworkers * 2):
            submit()
        while futs:
            fu = next(as_completed(list(futs)))
            k, s = futs.pop(fu)
            sid, ok, msg = fu.result()
            (done.add(sid) if ok else failed.__setitem__(sid, msg))
            if len(done) % 25 == 0:
                log(f"  fetched {len(done)}/{len(order)} shots, failed {len(failed)}")
            if time.time() < deadline:
                submit()
    final = {k: [s for s in v if (s in done or os.path.isdir(os.path.join(P["data"], f"{s}.zarr")))] for k, v in sel.items()}
    if time.time() >= deadline and any(len(final[k]) < len(sel[k]) for k in final):
        log(f"prep deadline reached: using the {sum(len(v) for v in final.values())} shots fetched so far "
            f"({sum(len(v) for v in sel.values()) - sum(len(v) for v in final.values())} still missing)")
    report2 = {k: dict(report[k], downloaded=len(final[k])) for k in final}
    meta = dict(split=split, shots=final, report=report2, failed=failed,
                config=dict(seed=CFG["seed"], task=CFG["task"], data_budget_gb=CFG["data_budget_gb"],
                            repo_id=CFG["repo_id"], git_revs=CFG["git_revs"], selected_at_min=round((time.time() - T0) / 60, 1)))
    json.dump(meta, open(P["sel"], "w"), indent=1)
    log(f"[{split}] FINAL shot counts: " + json.dumps({k: len(v) for k, v in final.items()}))
    if failed:
        log(f"  {len(failed)} shots failed and were excluded; first errors: {list(failed.items())[:3]}")
    if min(len(final["train"]), len(final["val"]), len(final["test"])) < 5:
        raise SystemExit("FATAL: fewer than 5 shots in some split; refusing to train on a misleading subset. "
                         "Check network/index logs above, extend TM_DATA_GB / TM_PREP_DEADLINE_H, and re-run prep.")


# ======================================================================================
# Datasets / model (mirrors tokamark_baseline/run_training.py exactly, minus subset_of_shots)
# ======================================================================================
def build_data(split, which, shuffle_windows=True):
    setup_imports()
    import torch
    from torch.utils.data import DataLoader
    from tokamark.tasks import get_task_config, get_task_metadata
    from tokamark.tools.transforms.compose_transform import ComposeTransforms
    from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset
    from src.model_transform import ModelTransform_1, ModelTransform_2
    from src.trainer import model_collate_fn
    from utils import seed_worker

    P = paths(split)
    sel = json.load(open(P["sel"]))
    assert sel["split"] == split, "selection file belongs to another split protocol!"
    _, stats, outl = split_files(split)
    patch_stats_module(stats)
    cfgt = get_task_config(task_name=CFG["task"])
    cfgt["stride_window"] = 0.005  # same override as run_training.py for task_1-x/2-x
    meta = get_task_metadata(config_task=cfgt, verbose=False)
    tf = ComposeTransforms([ModelTransform_1(meta | cfgt), ModelTransform_2(meta | cfgt)])
    nw = CFG["dl_workers"]
    if nw < 0:
        nw = max(2, min(8, os.cpu_count() or 2))
    smgr = {"base_local_zarr_path": P["data"]}
    loaders, g = {}, torch.Generator()
    g.manual_seed(CFG["seed"])
    # Tensor IPC must not go through /dev/shm (Docker default is 64 MB; a 512-window
    # batch deadlocks the workers silently). file_system uses /tmp instead.
    torch.multiprocessing.set_sharing_strategy("file_system")
    for w in which:
        mast = initialize_MAST_dataset(config_task=cfgt, shots_list=sel["shots"][w], local_flag=True, use_std_scaling=True,
                                       stats_metadata_file_path=stats, use_nan_filling=False, remove_outliers=True,
                                       outlier_metadata_file=outl, remove_bad_efit_rating=True,
                                       store_manager_settings=smgr, verbose=False)
        kw = dict(test_mode=True)
        if w == "test":
            kw["shuffle_windows"] = False
        ds = initialize_TokaMark_dataset(dataset=mast, task_metadata=meta, config_metadata=cfgt, custom_transform=tf, **kw)
        if ds is None:
            raise RuntimeError(f"{w} dataset empty after official filtering (outliers/bad EFIT)")
        try:
            t_probe = time.time()
            _ = ds[0]
            dt = time.time() - t_probe
            est = dt * CFG["batch_size"] / max(1, min(nw, CFG["batch_size"]))
            log(f"{w} dataset: {len(ds)} windows; first __getitem__ took {dt:.2f}s "
                f"(=> first batch of {CFG['batch_size']} will take roughly {est / 60:.1f} min)")
        except Exception as e:  # noqa
            log(f"{w} dataset probe failed (first item): {repr(e)[:200]}")
        extra = dict(drop_last=True) if w == "train" else {}
        if nw > 0:
            extra["persistent_workers"] = True
        loaders[w] = DataLoader(ds, batch_size=CFG["batch_size"], num_workers=nw, collate_fn=model_collate_fn,
                                worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                                prefetch_factor=2 if nw > 0 else None, **extra)
        log(f"{w} loader ready: {len(sel['shots'][w])} shots, batch_size={CFG['batch_size']}, workers={nw}")
    return loaders, meta | cfgt


def make_model(loader, meta):
    from src.multi_conv_mlp_model import create_cnn_architecture
    return create_cnn_architecture(dataloader_=loader, dict_metadata=meta, verbose=False)


# ======================================================================================
# TRAIN: resumable, deadline-aware, optional DataParallel (benchmarked)
# ======================================================================================
def atomic_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def pick_parallel(model, loader, dev, criterion, move):
    """Return (module_to_call, label). Benchmarks 1 GPU vs DataParallel on one cached batch."""
    import torch, torch.nn as nn
    n = torch.cuda.device_count()
    if n < 2 or CFG["use_dp"] == "0":
        return model, f"single GPU ({n} visible)"
    if CFG["use_dp"] == "1":
        return nn.DataParallel(model), f"DataParallel x{n} (forced)"
    try:
        batch = first_batch_timed(loader, tag="benchmark batch")
        x, y = move(batch, dev)
        def bench(m):
            opt = torch.optim.SGD(m.parameters(), lr=0.0)  # lr 0: benchmark does not change weights
            m.train()
            for i in range(14):
                if i == 4:
                    torch.cuda.synchronize(); t = time.time()
                opt.zero_grad(); criterion(m(*x), y).backward(); opt.step()
            torch.cuda.synchronize()
            return (time.time() - t) / 10
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        t1 = bench(model)
        dp = nn.DataParallel(model)
        t2 = bench(dp)
        model.load_state_dict(sd)
        log(f"GPU benchmark (compute only, same batch): 1 GPU {t1 * 1e3:.0f} ms/step, DataParallel {t2 * 1e3:.0f} ms/step")
        if t2 < t1 / 1.15:
            return dp, f"DataParallel x{n} (benchmark {t1 / t2:.2f}x faster)"
        return model, "single GPU (DataParallel not clearly faster)"
    except Exception as e:  # noqa
        log("parallel benchmark failed, using single GPU:", repr(e)[:200])
        return model, "single GPU (benchmark error)"


def cmd_train(split):
    setup_imports()
    import torch
    from src.trainer import MultiOutputMSELoss, move_batch_to_device
    from utils import set_seed
    set_seed(CFG["seed"])
    P = paths(split)
    os.makedirs(P["model_dir"], exist_ok=True)
    sel = json.load(open(P["sel"]))
    run_config = dict(task=CFG["task"], split=split, seed=CFG["seed"], batch_size=CFG["batch_size"],
                      lr=CFG["lr"], weight_decay=CFG["weight_decay"], max_steps=CFG["max_steps"],
                      validate_every=CFG["validate_every"], patience=CFG["patience"],
                      train_deadline_h=CFG["train_deadline_h"], use_dp=CFG["use_dp"],
                      git_revs=CFG["git_revs"],
                      shots={k: len(v) for k, v in sel["shots"].items()},
                      report=sel["report"])
    json.dump(run_config, open(os.path.join(P["out"], "run_config.json"), "w"), indent=1)
    log(f"run_config: {json.dumps(run_config)}")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"device={dev}, gpus={torch.cuda.device_count()}, cpus={os.cpu_count()}")
    phase = {"p": "building dataloaders", "step": 0}

    def _heartbeat():
        while True:
            time.sleep(60)
            log(f"  heartbeat: {phase['p']} (step {phase['step']})")

    threading.Thread(target=_heartbeat, daemon=True).start()
    loaders, meta = build_data(split, ["train", "val"])
    phase["p"] = "building model"
    t_model = time.time()
    log("building model (shape inference walks the dataset for one sample)...")
    model = make_model(loaders["train"], meta).to(dev)
    log(f"model built in {time.time() - t_model:.1f}s, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    crit = MultiOutputMSELoss()

    # ---- fail-fast smoke check (a few seconds): shapes, NaNs, forward/backward ----
    phase["p"] = "first batch (smoke)"
    b = first_batch_timed(loaders["train"], tag="first train batch")
    sid, wid, xs, ys = b
    x, y = move_batch_to_device(b, dev)
    out = model(*x)
    log("smoke: input shapes", [tuple(t.shape) for t in x], "| target shapes", [tuple(t.shape) for t in y],
        "| output shapes", [tuple(o.shape) for o in out])
    assert all(o.shape == t.shape for o, t in zip(out, y)), "output/target shape mismatch"
    assert all(torch.isfinite(t).all() for t in x), "non-finite inputs after collate"
    l = crit(out, y); l.backward()
    assert torch.isfinite(l), "non-finite initial loss"
    log(f"smoke OK: initial loss {l.item():.4f}")
    model.zero_grad(set_to_none=True)

    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    st = dict(step=0, best=float("inf"), no_improve=0, epoch=0, oom=0,
              hist=dict(step=[], train_loss=[], val_loss=[], lr=[], t=[]))
    last = os.path.join(P["model_dir"], "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); st = ck["state"]
        st.setdefault("oom", 0)
        st["hist"].setdefault("t", [])
        st["hist"].setdefault("lr", [])
        log(f"RESUMED from step {st['step']} (best val {st['best']:.5f})")
    runner, label = pick_parallel(model, loaders["train"], dev, crit, move_batch_to_device) if st["step"] == 0 else (model, "single GPU (resumed)")
    log("parallel mode:", label)

    @torch.no_grad()
    def validate():
        phase["p"] = "validation"
        runner.eval(); tot = n = 0
        for vb in loaders["val"]:
            if vb is None: continue
            vx, vy = move_batch_to_device(vb, dev)
            lv = crit(runner(*vx), vy); bs = vy[0].shape[0]
            tot += lv.item() * bs; n += bs
        phase["p"] = "training loop"
        if n == 0:
            raise RuntimeError("validation loader produced zero samples - check selection.json / val shots")
        return tot / n

    def save_last():
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), state=st), last)
        atomic_save(st["hist"], os.path.join(P["model_dir"], "training_history.pt"))
        json.dump(st["hist"], open(os.path.join(P["model_dir"], "training_history.json"), "w"))

    deadline = T0 + CFG["train_deadline_h"] * 3600
    phase["p"] = "training loop"
    it, run_loss, errs, bad, oom, t_last_ck, t_tr = iter(loaders["train"]), [], 0, 0, 0, time.time(), time.time()
    stop_reason = "max_steps"
    while st["step"] < CFG["max_steps"]:
        if time.time() > deadline:
            stop_reason = "time deadline"; break
        phase["p"] = f"waiting for batch (step {st['step']})"
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loaders["train"]); st["epoch"] += 1; run_loss = []; log(f"epoch {st['epoch']} done"); continue
        except Exception as e:  # noqa  (e.g. a corrupt shot inside a worker)
            errs += 1; log(f"data error #{errs}: {repr(e)[:200]}")
            if errs > 20: raise
            it = iter(loaders["train"]); continue
        if batch is None: continue
        phase["p"] = f"train step {st['step'] + 1}"
        try:
            runner.train()
            x, y = move_batch_to_device(batch, dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(runner(*x), y)
            if not torch.isfinite(loss):
                bad += 1; log(f"non-finite loss, skipping (#{bad})")
                if bad > 200: raise RuntimeError("too many non-finite losses")
                continue
            bad = 0
            loss.backward(); opt.step()
        except torch.cuda.OutOfMemoryError:
            oom += 1; st["oom"] = oom
            torch.cuda.empty_cache()
            log(f"CUDA OOM at step {st['step']} (#{oom}) - batch skipped; current batch_size={CFG['batch_size']}")
            if oom >= 3:
                raise RuntimeError(
                    f"CUDA out of memory {oom}x with batch_size={CFG['batch_size']} on this GPU. "
                    "Re-run the train cell with a smaller TM_BATCH in the config cell (e.g. 256 or 128); "
                    f"training will resume from {last} without losing progress.")
            continue
        st["step"] += 1; run_loss.append(loss.item()); phase["step"] = st["step"]
        if st["step"] <= 10 or st["step"] % 20 == 0:
            el = time.time() - t_tr; t_tr = time.time()
            log(f"step {st['step']} loss {sum(run_loss) / len(run_loss):.4f} "
                f"({(CFG['batch_size'] if st['step'] <= 10 else 20 * CFG['batch_size']) / el:.0f} samples/s)")
        if st["step"] == 1:
            save_last()  # keep progress even if the session dies early
        if st["step"] % CFG["validate_every"] == 0:
            v = validate(); h = st["hist"]
            for k, val in zip(("step", "train_loss", "val_loss", "lr", "t"), (st["step"], sum(run_loss) / len(run_loss), v, CFG["lr"], time.time() - T0)):
                h[k].append(val)
            if v < st["best"]:
                st["best"], st["no_improve"] = v, 0
                atomic_save(model.state_dict(), os.path.join(P["model_dir"], "best_model.pt"))
            else:
                st["no_improve"] += 1
            log(f"VAL step {st['step']} val_loss {v:.5f} best {st['best']:.5f} no_improve {st['no_improve']}/{CFG['patience']}")
            save_last(); t_last_ck = time.time()
            if st["no_improve"] >= CFG["patience"]:
                stop_reason = "early stopping (official patience rule)"; break
        elif time.time() - t_last_ck > CFG["ckpt_every_s"]:
            save_last(); t_last_ck = time.time()
    v = validate()
    if v < st["best"] or not os.path.exists(os.path.join(P["model_dir"], "best_model.pt")):
        st["best"] = v; atomic_save(model.state_dict(), os.path.join(P["model_dir"], "best_model.pt"))
    save_last()
    summary = dict(stop_reason=stop_reason, steps=st["step"], epochs=st["epoch"], best_val=st["best"],
                   parallel=label, cuda_oom_events=st.get("oom", 0), data_errors=errs,
                   duration_min=round((time.time() - T0) / 60, 1),
                   model_dir=P["model_dir"], **{k: run_config[k] for k in ("task", "split", "seed", "batch_size", "lr",
                                                                           "weight_decay", "validate_every", "patience",
                                                                           "git_revs", "shots", "report")})
    json.dump(summary, open(os.path.join(P["out"], "train_summary.json"), "w"), indent=1)
    log(f"TRAIN DONE ({stop_reason}) at step {st['step']}, best val {st['best']:.5f}; summary -> train_summary.json")


# ======================================================================================
# EVAL: official evaluator on test shots, un-standardized NRMSE
# ======================================================================================
def cmd_eval(split):
    setup_imports()
    import torch
    from tokamark.evaluator import WindowMetricsAccumulator, compute_metrics
    from src.evaluator import cnn_unstd_evaluation_per_shot
    P = paths(split)
    os.makedirs(P["results"], exist_ok=True)
    if not os.path.exists(os.path.join(P["model_dir"], "best_model.pt")):
        raise SystemExit(f"FATAL: {P['model_dir']}best_model.pt not found - run train first")
    loaders, meta = build_data(split, ["test"])
    model = make_model(loaders["test"], meta)
    acc = WindowMetricsAccumulator(CFG["task"])
    cnn_unstd_evaluation_per_shot(test_dataloader=loaders["test"], config_task=meta, cnn_model=model,
                                  output_dir=P["model_dir"], accumulator=acc)
    compute_metrics(task=CFG["task"], output_dir=P["results"], window_metrics_accumulator=acc,
                    save_windows_metrics=True, save_task_metrics=True)
    sel = json.load(open(P["sel"]))
    log(f"[{split}] EVAL DONE. Results in {P['results']}")
    log(f"[{split}] shot counts used: {json.dumps(sel['report'])}")
    for f in sorted(os.listdir(P["results"])):
        if f.endswith((".json", ".yaml", ".yml")):
            print(f"--- {f}\n" + open(os.path.join(P["results"], f)).read()[:3000])
    for root, _, files in os.walk(P["results"]):
        for f in sorted(files):
            if f.endswith(".csv"):
                print(f"--- {os.path.join(root, f)}")
                with open(os.path.join(root, f)) as fh:
                    for i, line in enumerate(fh):
                        if i > 12: print("  ..."); break
                        print("  " + line.rstrip())


# ======================================================================================
# PLOT: history curves + optional reconstruction-quality viz (never blocks core results)
# ======================================================================================
def cmd_plot(split):
    setup_imports()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = paths(split)
    os.makedirs(P["figures"], exist_ok=True)

    # (a) training/validation curves - core, from saved history
    hp = os.path.join(P["model_dir"], "training_history.pt")
    try:
        h = torch.load(hp, map_location="cpu", weights_only=False)
        steps, tr_, va_ = h["step"], h["train_loss"], h["val_loss"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, tr_, label="train loss (running mean at validation)")
        ax.plot(steps, va_, label="validation loss")
        ax.set_xlabel("step"); ax.set_ylabel("MultiOutputMSELoss (standardised)")
        ax.set_title(f"TokaMark {CFG['task']} baseline - {split} split, seed {CFG['seed']}")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(P["figures"], "loss_curves.png"), dpi=120)
        plt.close(fig)
        log(f"saved loss_curves.png ({len(steps)} validation points)")
    except Exception as e:  # noqa
        log("loss-curve plot failed:", repr(e)[:200])

    # (b) reconstruction-quality viz: predicted vs true psi on a few test windows
    try:
        from src.trainer import move_batch_to_device
        from src.evaluator import cnn_unstd_evaluation_per_shot  # noqa: F401 (kept for parity)
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loaders, meta = build_data(split, ["test"])
        model = make_model(loaders["test"], meta).to(dev)
        model.load_state_dict(torch.load(os.path.join(P["model_dir"], "best_model.pt"), map_location=dev, weights_only=False))
        model.eval()
        batch = first_batch(loaders["test"])
        shot_ids, win_ids, x, y = move_batch_to_device(batch, dev)
        with torch.no_grad():
            y_pred = model(*x)
        mstd = meta["output"]["equilibrium-psi"]
        mu, sd = float(mstd["mean"]), float(mstd["std"])
        n = min(CFG["plot_windows"], y[0].shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
        if n == 1:
            axes = [axes]
        for i in range(n):
            true = (y[0][i, 0].cpu().numpy() * sd + mu)
            pred = (y_pred[0][i, 0].cpu().numpy() * sd + mu)
            for j, (img, ttl) in enumerate(((true, "true psi"), (pred, "pred psi"), (abs(pred - true), "|error|"))):
                im = axes[i][j].imshow(img, cmap="viridis" if j < 2 else "magma")
                axes[i][j].set_title(f"{ttl} (shot {int(shot_ids[i])}, win {int(win_ids[i])})", fontsize=9)
                axes[i][j].axis("off")
                fig.colorbar(im, ax=axes[i][j], fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(P["figures"], "psi_reconstruction_samples.png"), dpi=120)
        plt.close(fig)
        log("saved psi_reconstruction_samples.png")
    except Exception as e:  # noqa
        log("optional psi reconstruction viz failed (core results unaffected):", repr(e)[:300])
    log(f"figures in {P['figures']}")


# ======================================================================================
# SUMMARY: one human-readable recap of the whole run
# ======================================================================================
def cmd_summary(split):
    setup_imports()
    import csv as _csv
    P = paths(split)
    lines = [f"# TokaMark {CFG['task']} baseline run - {split} split", ""]
    def add_json(fname, title):
        p = os.path.join(P["out"], fname)
        if os.path.exists(p):
            lines.append(f"## {title}")
            lines.append("```json")
            lines.append(json.dumps(json.load(open(p)), indent=1))
            lines.append("```")
        else:
            lines.append(f"## {title}\n\nMISSING: {p}")
        lines.append("")
    add_json("selection.json", "Data subset & split (official membership, budget-limited selection)")
    add_json("run_config.json", "Run configuration")
    add_json("train_summary.json", "Training summary")
    lines.append("## Evaluation (official evaluator, un-standardised NRMSE)")
    tm = os.path.join(P["results"], CFG["task"], "task_metrics.csv")
    if os.path.exists(tm):
        rows = list(_csv.DictReader(open(tm)))
        for r in rows:
            name = r["feature_name"]
            lines.append(f"- **{name}**: NRMSE={float(r['NRMSE_mean']):.4f} (+/-{float(r['NRMSE_std_pop']):.4f}), "
                         f"RMSE={float(r['RMSE_mean']):.5f}, MAE={float(r['MAE_mean']):.5f}, n_shots={r['n_shots']}")
    else:
        lines.append(f"MISSING: {tm} (eval not run?)")
    lines.append("")
    lines.append("## Artifacts")
    for label, p in (("selection", P["sel"]), ("run config", os.path.join(P["out"], "run_config.json")),
                     ("train summary", os.path.join(P["out"], "train_summary.json")),
                     ("best model", os.path.join(P["model_dir"], "best_model.pt")),
                     ("resume checkpoint", os.path.join(P["model_dir"], "last.pt")),
                     ("history", os.path.join(P["model_dir"], "training_history.pt")),
                     ("results dir", P["results"]), ("figures dir", P["figures"])):
        lines.append(f"- {label}: `{p}` {'OK' if os.path.exists(p) else '**MISSING**'}")
    text = "\n".join(lines)
    with open(os.path.join(P["out"], "summary.md"), "w") as fh:
        fh.write(text)
    print(text)
    log(f"summary -> {os.path.join(P['out'], 'summary.md')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval", "plot", "summary"])
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval, "plot": cmd_plot, "summary": cmd_summary}[a.cmd](a.split)
