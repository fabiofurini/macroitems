"""Readers for the instance formats used in the literature.

Three external formats are supported, all mapped onto the conventions of this
package -- arcs ``(i, j)`` meaning "j is a prerequisite of i" (so ``x_i <= x_j``),
positive weights, profits of arbitrary sign:

``read_pckp_dat``
    The benchmark of the precedence-constrained knapsack literature (Park and
    Park 1997; Boland, Bley, Fricke, Froyland, Sotirov, *Math. Programming*
    2012; Espinoza, Goycoolea, Moreno, Newman 2015): 23 instances in two
    families, *mining* (A-K, sparse) and *telecom* (L-W, dense).  The
    ``.dat`` file carries the capacity, so these instances come with the
    capacity their authors used.
``read_pckp_lp``
    The same instances in CPLEX LP format, read back into an
    :class:`~macroitems.instance.Instance`.  Slower than the ``.dat`` reader
    and provided mainly to check it.
``read_minelib_upit``
    MineLib (Espinoza, Goycoolea, Moreno, Newman, *Ann. Oper. Res.* 2013):
    the ``.blocks``/``.prec``/``.upit`` triple of an ultimate-pit instance.

None of these datasets is redistributed with this package; the readers take
the files as published by their authors.  See ``docs/instances.md``.
"""
from __future__ import annotations

import os
import re
import zipfile
from typing import Optional

import numpy as np

from .instance import Instance

__all__ = ["read_pckp_dat", "read_pckp_lp", "read_minelib_upit", "read_any"]


# --------------------------------------------------------------- PCKP .dat
def read_pckp_dat(path: str, name: Optional[str] = None) -> Instance:
    """Read a precedence-constrained knapsack instance in ``.dat`` format.

    Layout (tab-separated)::

        n  m  capacity
        id  profit  weight  n_pred  <empty>  pred_1 ... pred_{n_pred}
        ...

    ``pred_r`` are the prerequisites of ``id``, giving arcs ``(id, pred_r)``.
    The capacity of the header is stored in ``inst.meta["capacity"]``.
    """
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    head = lines[0].split()
    n, m = int(head[0]), int(head[1])
    capacity = float(head[2]) if len(head) > 2 else None

    p = np.zeros(n)
    w = np.zeros(n)
    arcs: list[tuple[int, int]] = []
    seen = np.zeros(n, dtype=bool)
    for ln in lines[1:]:
        f_ = ln.split()
        if len(f_) < 4:
            continue
        i = int(f_[0])
        if not 0 <= i < n:
            raise ValueError(f"{path}: item index {i} out of range 0..{n - 1}")
        p[i], w[i] = float(f_[1]), float(f_[2])
        npred = int(f_[3])
        preds = [int(v) for v in f_[4:4 + npred]]
        if len(preds) != npred:
            raise ValueError(f"{path}: item {i} declares {npred} predecessors, found {len(preds)}")
        arcs.extend((i, j) for j in preds)
        seen[i] = True
    if not seen.all():
        raise ValueError(f"{path}: {int((~seen).sum())} items missing from the file")
    if len(arcs) != m:
        raise ValueError(f"{path}: header declares {m} arcs, found {len(arcs)}")

    meta = {"source": "PCKP benchmark (.dat)", "file": os.path.basename(path)}
    if capacity is not None:
        meta["capacity"] = capacity
    meta["family"] = _pckp_family(os.path.basename(path))
    inst = Instance(p, w, np.array(arcs, dtype=np.int64) if arcs else np.zeros((0, 2), np.int64),
                    name=name or _strip_suffixes(os.path.basename(path)), meta=meta)
    inst.validate()
    return inst


def _pckp_family(filename: str) -> str:
    """Family of a benchmark instance, from the name.

    The three families of the published distribution are ``telecom`` (A-K),
    ``mining`` (L-W) and ``scheduling`` (files named ``n_m_k``).  Note that the
    ``runList_*`` files shipped with some copies of the benchmark label A-K as
    "mining" and L-W as "telecom", which is the opposite of the directory
    layout and of ``results.txt`` in the original distribution; the original
    is followed here, and it agrees with the data (the mining instances are
    the ones with negative-profit waste blocks).
    """
    letter = filename[:1].upper()
    if letter.isdigit():
        return "scheduling"
    if "A" <= letter <= "K":
        return "telecom"
    if "L" <= letter <= "W":
        return "mining"
    return "unknown"


def _strip_suffixes(filename: str) -> str:
    for suffix in (".lp.dat", ".dat", ".lp"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


# ---------------------------------------------------------------- PCKP .lp
_LP_TERM = re.compile(r"([+-]?)\s*(\d*\.?\d*(?:[eE][+-]?\d+)?)\s*x(\d+)")


def read_pckp_lp(path: str, name: Optional[str] = None) -> Instance:
    """Read the same instances from CPLEX LP format.

    The file is expected to hold one maximization objective, one knapsack row
    (the only constraint with more than two terms), and one precedence row
    ``x_i - x_j <= 0`` per arc.  Variables are ``x1..xn`` (1-based).
    """
    with open(path) as f:
        text = f.read()
    sections = _split_lp_sections(text)
    obj_terms = _lp_terms(sections["objective"])
    n = max(idx for idx, _ in obj_terms)
    p = np.zeros(n)
    for idx, coef in obj_terms:
        p[idx - 1] = coef

    w = np.zeros(n)
    arcs: list[tuple[int, int]] = []
    capacity = None
    for body, rhs in _lp_constraints(sections["constraints"]):
        terms = _lp_terms(body)
        if len(terms) > 2:                       # the knapsack row
            for idx, coef in terms:
                w[idx - 1] = coef
            capacity = rhs
        elif len(terms) == 2:                    # x_i - x_j <= 0
            (i, ci), (j, cj) = terms
            if ci > 0 and cj < 0:
                arcs.append((i - 1, j - 1))
            elif cj > 0 and ci < 0:
                arcs.append((j - 1, i - 1))
            else:
                raise ValueError(f"{path}: constraint with two terms of equal sign: {body!r}")
    meta = {"source": "PCKP benchmark (.lp)", "file": os.path.basename(path),
            "family": _pckp_family(os.path.basename(path))}
    if capacity is not None:
        meta["capacity"] = float(capacity)
    inst = Instance(p, w, np.array(arcs, dtype=np.int64) if arcs else np.zeros((0, 2), np.int64),
                    name=name or _strip_suffixes(os.path.basename(path)), meta=meta)
    inst.validate()
    return inst


def _split_lp_sections(text: str) -> dict:
    low = text.lower()
    i_obj = low.index("maximize") if "maximize" in low else low.index("minimize")
    i_st = min(k for k in (low.find("subject to"), low.find("st\n"), low.find("s.t.")) if k > 0)
    i_end = min(k for k in (low.find("\nbounds"), low.find("\nbinaries"), low.find("\nbinary"),
                            low.find("\ngeneral"), low.find("\nend")) if k > 0)
    objective = text[i_obj:i_st]
    objective = objective[objective.index(":") + 1:] if ":" in objective.split("\n", 1)[0] + ":" else objective
    return {"objective": objective, "constraints": text[i_st:i_end]}


def _lp_terms(body: str) -> list[tuple[int, float]]:
    terms = []
    for sign, num, idx in _LP_TERM.findall(body):
        coef = float(num) if num not in ("", ".") else 1.0
        terms.append((int(idx), -coef if sign == "-" else coef))
    return terms


def _lp_constraints(section: str):
    section = section.split("\n", 1)[1] if section.lower().startswith("subject to") else section
    # a constraint starts at a name "cNN:" and ends at the next one
    parts = re.split(r"\n\s*[A-Za-z_][A-Za-z0-9_]*\s*:", "\n" + section)
    for part in parts[1:]:
        mobj = re.search(r"(<=|>=|=<|=>|=)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", part)
        if mobj is None:
            continue
        yield part[: mobj.start()], float(mobj.group(2))


# ---------------------------------------------------------------- MineLib
def read_minelib_upit(source: str, weight: str = "auto", resource: int = 0,
                      tonnage_column: Optional[int] = None,
                      name: Optional[str] = None) -> Instance:
    """Read a MineLib instance as a single-capacity PCKP instance.

    ``source`` is a directory, a zip archive, or the stem of an instance
    (``.../minelib/newman1``); the ``.upit``, ``.prec``, ``.blocks`` and
    ``.cpit`` files are picked up from it.

    Profits are the UPIT objective coefficients.  The weight is chosen by
    ``weight``:

    ``"cpit"`` (default)
        the operational-resource coefficient :math:`q_{br}` of resource
        ``resource`` in the ``.cpit`` file.  This is the amount of resource
        needed to extract the block -- the tonnage, for the mining resource --
        as *declared by the instance itself*, which avoids having to guess
        which ``.blocks`` column holds it (the attribute layout differs across
        instances).  The per-period limit :math:`\\bar R_{r}` is stored in
        ``meta["capacity_cpit_period"]``, giving a capacity taken from the
        literature rather than invented.
    ``"blocks"``
        the ``.blocks`` attribute in ``tonnage_column`` (0-based among the
        attributes that follow ``id x y z``).
    ``"unit"``
        unit weights, i.e. the volume parameterization of Lerchs and Grossmann.

    A ``.prec`` line ``b k b_1 ... b_k`` says that the blocks ``b_r`` must be
    extracted before ``b``, giving arcs ``(b, b_r)`` -- this package's
    convention already.
    """
    files = _minelib_files(source)
    if ".upit" not in files or ".prec" not in files:
        raise ValueError(f"{source}: need both a .upit and a .prec file, found {sorted(files)}")

    vals, nblocks, in_obj = {}, None, False
    for ln in files[".upit"]:
        s = ln.strip()
        if not s:
            continue
        upper = s.upper()
        if upper.startswith("NBLOCKS"):
            nblocks = int(s.split(":")[1])
        if upper.startswith("OBJECTIVE_FUNCTION"):
            in_obj = True
            continue
        if in_obj:
            parts = s.split()
            if parts[0].upper().startswith("EOF"):
                break
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                vals[int(parts[0])] = float(parts[1])
    n = nblocks if nblocks is not None else (max(vals) + 1)
    p = np.zeros(n)
    for b, v in vals.items():
        p[b] = v

    arcs = []
    for ln in files[".prec"]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        b, k = int(parts[0]), int(parts[1])
        arcs.extend((b, int(r)) for r in parts[2:2 + k])

    meta = {"source": "MineLib", "file": os.path.basename(source)}
    w = np.ones(n)
    if weight == "auto":
        weight, resource, tonnage_column, why = _resolve_minelib_weight(files, n, resource)
        meta["weight_resolution"] = why
    if weight == "cpit":
        if ".cpit" not in files:
            raise ValueError(f"{source}: weight='cpit' needs the .cpit file")
        coeffs, limits = _minelib_cpit_resources(files[".cpit"], resource)
        missing = [b for b in range(n) if b not in coeffs]
        if missing:
            raise ValueError(f"{source}: resource {resource} has no coefficient for "
                             f"{len(missing)} blocks (first: {missing[0]})")
        for b, v in coeffs.items():
            w[b] = v
        meta["weight"] = f"cpit resource {resource} coefficient (q_br)"
        if limits:
            meta["capacity_cpit_period"] = float(min(limits))
            meta["capacity_cpit_total"] = float(sum(limits))
            meta["cpit_n_periods"] = len(limits)
    elif weight == "blocks":
        if ".blocks" not in files or tonnage_column is None:
            raise ValueError(f"{source}: weight='blocks' needs the .blocks file and tonnage_column")
        for ln in files[".blocks"]:
            parts = ln.split()
            if len(parts) < 5 + tonnage_column:
                continue
            w[int(parts[0])] = float(parts[4 + tonnage_column])
        meta["weight"] = f"blocks attribute column {tonnage_column}"
    elif weight == "unit":
        meta["weight"] = "unit weights (volume parameterization)"
    else:
        raise ValueError(f"weight must be 'cpit', 'blocks' or 'unit', got {weight!r}")
    if not np.all(w > 0):
        bad = int((w <= 0).sum())
        raise ValueError(f"{source}: {bad} blocks have non-positive weight under weight={weight!r}")

    inst = Instance(p, w, np.array(arcs, dtype=np.int64) if arcs else np.zeros((0, 2), np.int64),
                    name=name or _strip_suffixes(os.path.basename(source)), meta=meta)
    inst.validate()
    return inst


def _resolve_minelib_weight(files: dict, n: int, resource: int):
    """Find the tonnage of a MineLib instance without per-instance configuration.

    Every block has to be moved, so the tonnage is a quantity that is positive
    on *all* blocks.  Two places declare it, and they are cross-checked against
    each other:

    1. an operational resource of the ``.cpit`` file whose coefficients cover
       all blocks -- the mining resource (a resource covering only part of the
       blocks is a processing resource, which ore blocks alone consume);
    2. failing that, the ``.blocks`` attribute that is positive on every block
       *and* agrees with the coefficients of a partial ``.cpit`` resource
       wherever those are defined.  The agreement is what identifies the
       column: a processing resource charges ore blocks exactly their tonnage.

    Returns ``(weight, resource, tonnage_column, explanation)``.
    """
    if ".cpit" in files:
        for r in range(_minelib_n_resources(files[".cpit"])):
            coeffs, _ = _minelib_cpit_resources(files[".cpit"], r)
            if len(coeffs) == n and all(v > 0 for v in coeffs.values()):
                return "cpit", r, None, f"cpit resource {r} covers all {n} blocks"

    if ".blocks" in files:
        attrs: dict[int, list[float]] = {}
        for ln in files[".blocks"]:
            parts = ln.split()
            if len(parts) > 4:
                try:
                    attrs[int(parts[0])] = [float(x) for x in parts[4:]]
                except ValueError:      # non-numeric attributes (rock type, ...)
                    attrs[int(parts[0])] = [float(x) if _is_number(x) else float("nan")
                                            for x in parts[4:]]
        if len(attrs) == n:
            ncol = min(len(v) for v in attrs.values())
            reference: dict[int, float] = {}
            if ".cpit" in files:
                for r in range(_minelib_n_resources(files[".cpit"])):
                    coeffs, _ = _minelib_cpit_resources(files[".cpit"], r)
                    if coeffs:
                        reference = coeffs
                        break
            for k in range(ncol):
                column = {b: v[k] for b, v in attrs.items()}
                if not all(x > 0 for x in column.values()):
                    continue
                if reference:
                    if all(abs(column[b] - v) <= 1e-6 * max(1.0, abs(v))
                           for b, v in reference.items() if b in column):
                        return ("blocks", resource, k,
                                f"blocks attribute {k}: positive on all blocks and equal to the "
                                f"cpit resource coefficients on the {len(reference)} blocks that "
                                f"declare one")
                else:
                    return "blocks", resource, k, f"blocks attribute {k}: positive on all blocks"

    return "unit", resource, None, "no all-block tonnage found: unit weights (volume)"


def _minelib_n_resources(lines: list[str]) -> int:
    for ln in lines:
        if ln.upper().startswith("NRESOURCE"):
            try:
                return int(ln.split(":")[1])
            except (IndexError, ValueError):
                break
    return 3


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _minelib_files(source: str) -> dict:
    """Collect the MineLib files of an instance from a dir, a zip, or a stem."""
    exts = (".upit", ".prec", ".blocks", ".cpit")
    files: dict[str, list[str]] = {}
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            for fn in z.namelist():
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    files[ext] = z.read(fn).decode("utf-8", errors="replace").splitlines()
        return files
    if os.path.isdir(source):
        for fn in os.listdir(source):
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                with open(os.path.join(source, fn)) as f:
                    files[ext] = f.read().splitlines()
        return files
    # a stem such as ".../minelib/newman1"
    for ext in exts:
        path = source + ext
        if os.path.exists(path):
            with open(path) as f:
                files[ext] = f.read().splitlines()
    if not files:
        raise FileNotFoundError(f"{source}: no MineLib files found (dir, zip or instance stem)")
    return files


def _minelib_cpit_resources(lines: list[str], resource: int):
    """Return ``(coefficients, limits)`` for one operational resource of a .cpit file.

    ``RESOURCE_CONSTRAINT_COEFFICIENTS`` lines are ``b r v`` (or ``b d r v``
    when destinations are present) and ``RESOURCE_CONSTRAINT_LIMITS`` lines are
    ``r t c v`` with ``c`` in ``L``/``G``/``I``; only upper limits (``L``, and
    the second value of ``I``) are collected.
    """
    coeffs: dict[int, float] = {}
    limits: list[float] = []
    section = None
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        upper = s.upper()
        if upper.startswith("RESOURCE_CONSTRAINT_COEFFICIENTS"):
            section = "coef"
            continue
        if upper.startswith("RESOURCE_CONSTRAINT_LIMITS"):
            section = "lim"
            continue
        if upper.endswith(":") or (":" in s and not s[0].isdigit()):
            section = None
            continue
        parts = s.split()
        if section == "coef":
            if len(parts) == 3:                       # b r v
                b, r, v = int(parts[0]), int(parts[1]), float(parts[2])
            elif len(parts) == 4:                     # b d r v
                b, r, v = int(parts[0]), int(parts[2]), float(parts[3])
            else:
                continue
            if r == resource:
                coeffs[b] = v
        elif section == "lim":
            if len(parts) >= 4 and int(parts[0]) == resource:
                kind = parts[2].upper()
                if kind == "L":
                    limits.append(float(parts[3]))
                elif kind == "I" and len(parts) >= 5:
                    limits.append(float(parts[4]))
    return coeffs, limits


# -------------------------------------------------------------- dispatcher
def read_any(path: str, **kwargs) -> Instance:
    """Read an instance, guessing the format from the name."""
    low = path.lower()
    if low.endswith(".dat"):
        return read_pckp_dat(path, **kwargs)
    if low.endswith(".lp"):
        return read_pckp_lp(path, **kwargs)
    if os.path.isdir(path) or zipfile.is_zipfile(path):
        return read_minelib_upit(path, **kwargs)
    return Instance.read(path)
