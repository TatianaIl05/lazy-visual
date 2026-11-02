from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ._errors import err_export_format, err_no_write_permission

VALID_FMTS = {"png", "svg", "pdf"}

def _resolve_targets(path, fmt):
    items = list(path) if isinstance(path, (list, tuple)) else [path]
    out = []
    for p in items:
        p = Path(p)
        f = (fmt or p.suffix.lstrip(".") or "png").lower()
        if f not in VALID_FMTS:
            raise err_export_format(f)
        if p.suffix.lower() != f".{f}":
            p = p.with_suffix(f".{f}")
        out.append((p, f))
    return out

def _atomic_save(fig, target, fmt, dpi):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=f".{fmt}", dir=str(target.parent))
    os.close(fd)
    try:
        extra = [a.get_legend() for a in fig.axes if a.get_legend()]
        if extra:
            fig.savefig(tmp, format=fmt, dpi=dpi,
                        bbox_inches="tight",
                        bbox_extra_artists=extra)
        else:
            fig.savefig(tmp, format=fmt, dpi=dpi, bbox_inches="tight")
        os.replace(tmp, target)
    except PermissionError as e:
        try: os.unlink(tmp)
        except OSError: pass
        raise err_no_write_permission(str(target)) from e
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def save_result(fig, path, *, fmt=None, dpi=None):
    targets = _resolve_targets(path, fmt)
    files = []
    for target, f in targets:
        _atomic_save(fig, target, f, dpi)
        files.append(target)
    return files
