from __future__ import annotations

import difflib
from collections.abc import Iterable


class LvError(Exception):
    code: str = "LV-E000"

    def __init__(self, message: str, *, code: str | None = None,
                 hint: str | None = None) -> None:
        self.code = code or self.code
        self.message = message
        self.hint = hint
        super().__init__(self._format())

    def _format(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.hint:
            base += f" Hint: {self.hint}"
        return base

    def __str__(self) -> str:
        return self._format()


class LvDataError(LvError, ValueError):
    code = "LV-E100"


class LvArgumentError(LvError, ValueError):
    code = "LV-E200"


class LvExportError(LvError, OSError):
    code = "LV-E300"


class LvDependencyError(LvError, ImportError):
    code = "LV-E401"


class LvDataWarning(UserWarning):
    pass


class LvPerformanceWarning(UserWarning):
    pass


def _suggest(name: str, available: Iterable[str]) -> str | None:
    m = difflib.get_close_matches(name, list(available), n=1, cutoff=0.6)
    return m[0] if m else None


def err_column_not_found(name: str, available: list) -> LvDataError:
    hint = "column names are case-sensitive"
    s = _suggest(name, available)
    if s:
        hint = f"did you mean '{s}'? {hint}"
    return LvDataError(
        f"Column '{name}' not found. Available: {', '.join(available)}.",
        code="LV-E101", hint=hint)


def err_wrong_dtype(name: str, expected: str, found: str) -> LvDataError:
    return LvDataError(
        f"Column '{name}' has invalid dtype '{found}', expected {expected}.",
        code="LV-E102")


def err_empty_after_nan(n_before: int, n_dropped: int) -> LvDataError:
    return LvDataError(
        f"No rows left after dropping NaNs "
        f"(had {n_before}, dropped {n_dropped}).", code="LV-E103")


def err_duplicate_columns(dupes: list) -> LvDataError:
    return LvDataError(
        f"Duplicate column names: {', '.join(dupes)}.",
        code="LV-E104", hint="rename columns to be unique")


def err_y_true_not_binary(classes: list) -> LvDataError:
    return LvDataError(
        f"y_true must have exactly 2 classes, found {len(classes)}: "
        f"{classes}.", code="LV-E106")


def err_upset_sets_count(n: int) -> LvDataError:
    return LvDataError(f"upset requires 2-20 sets, got {n}.",
                       code="LV-E107")


def err_file_not_found(path: str) -> LvDataError:
    return LvDataError(f"File not found or unreadable: {path}.",
                       code="LV-E108")


def err_unsupported_format(fmt: str) -> LvDataError:
    return LvDataError(f"Unsupported file format: '{fmt}'.",
                       code="LV-E109")


def err_url_unreachable(url: str, reason: str) -> LvDataError:
    return LvDataError(
        f"URL unreachable or timed out: {url} ({reason}).",
        code="LV-E110")


def err_out_of_range(name: str, value: object,
                     allowed: str) -> LvArgumentError:
    return LvArgumentError(
        f"Argument '{name}' = {value!r} is out of range. "
        f"Allowed: {allowed}.", code="LV-E201")


def err_wrong_arg_type(name: str, expected: str,
                       got: str) -> LvArgumentError:
    return LvArgumentError(
        f"Argument '{name}' must be {expected}, got {got}.",
        code="LV-E202")


def err_unknown_theme(name: str, available: list) -> LvArgumentError:
    return LvArgumentError(
        f"Unknown theme/palette '{name}'. "
        f"Available: {', '.join(available)}.", code="LV-E203")


def err_incompatible_args(message: str) -> LvArgumentError:
    return LvArgumentError(message, code="LV-E204")


def err_too_many_groups(n: int, limit: int) -> LvDataError:
    return LvDataError(
        f"Too many groups on axis ({n} > {limit}).",
        code="LV-E205", hint="reduce the number of categories")


def err_export_format(fmt: str) -> LvExportError:
    return LvExportError(
        f"Unsupported export format '{fmt}'. Use png/svg/pdf.",
        code="LV-E301")


def err_no_write_permission(path: str) -> LvExportError:
    return LvExportError(f"Cannot write to {path}.", code="LV-E302")


def err_backend_unavailable(backend: str) -> LvArgumentError:
    return LvArgumentError(
        f"Backend '{backend}' is not available in this version.",
        code="LV-E310", hint="only 'matplotlib' is supported in 0.1.0")


def err_missing_dependency(pkg: str, extra: str) -> LvDependencyError:
    return LvDependencyError(
        f"Optional dependency '{pkg}' is missing. "
        f"Install with: pip install lazy-visual[{extra}]",
        code="LV-E401")
