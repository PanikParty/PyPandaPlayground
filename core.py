"""DataFrame Workbench engine. Runs inside Pyodide; every public call goes
through dispatch(cmd, json_args) and returns a JSON string."""
import ast
import contextlib
import io
import json
import math
import os
import re
import traceback

import numpy as np
import pandas as pd

NS = {"pd": pd, "np": np}
_VIEW_CACHE = {}
ENGINE_FILE = "<workbench>"


# ---------------------------------------------------------------- helpers
def _label(c):
    return c if isinstance(c, str) else str(c)


def _kind(dtype):
    t = pd.api.types
    if dtype == object:
        return "object"
    if t.is_bool_dtype(dtype):
        return "bool"
    if t.is_integer_dtype(dtype):
        return "int"
    if t.is_float_dtype(dtype) or t.is_complex_dtype(dtype):
        return "float"
    if t.is_datetime64_any_dtype(dtype):
        return "datetime"
    if t.is_timedelta64_dtype(dtype):
        return "timedelta"
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if t.is_string_dtype(dtype):
        return "string"
    return "object"


def _is_numeric(s):
    return _kind(s.dtype) in ("int", "float")


def _ffloat(v):
    v = float(v)
    if not math.isfinite(v):
        return "inf" if v > 0 else ("-inf" if v < 0 else None)
    a = abs(v)
    if a == 0 or (1e-4 <= a < 1e15):
        s = f"{v:.6f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    return f"{v:.6e}"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fcell(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return _ffloat(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    s = str(v)
    return s if len(s) <= 300 else s[:297] + "…"


def _fmt_series(s):
    k = _kind(s.dtype)
    mask = s.isna().to_numpy()
    if k == "datetime":
        midnight = bool((s.dropna().dt.normalize() == s.dropna()).all())
        txt = s.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S")
        return [None if m else t for t, m in zip(txt.tolist(), mask)]
    vals = s.tolist()
    return [None if m else _fcell(v) for v, m in zip(vals, mask)]


def _fmt_index(idx):
    if isinstance(idx, pd.MultiIndex):
        return [" · ".join(str(p) for p in t) for t in idx]
    return [("" if v is None else v) for v in _fmt_series(pd.Series(idx))]


def _table(d, max_rows=None):
    if max_rows is not None:
        d = d.head(max_rows)
    cols = []
    data = []
    for i in range(d.shape[1]):
        s = d.iloc[:, i]
        cols.append({"label": _label(d.columns[i]), "dtype": str(s.dtype), "kind": _kind(s.dtype)})
        data.append(_fmt_series(s))
    return {
        "columns": cols,
        "data": data,
        "index": _fmt_index(d.index),
        "index_name": None if d.index.name is None else str(d.index.name),
    }


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _num(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _frame(name):
    if name not in NS or not isinstance(NS[name], pd.DataFrame):
        raise LookupError(f"No DataFrame named {name!r}. It may have been deleted or reassigned in the console.")
    return NS[name]


def frames():
    return [
        {"name": k, "rows": int(v.shape[0]), "cols": int(v.shape[1])}
        for k, v in NS.items()
        if not k.startswith("_") and isinstance(v, pd.DataFrame)
    ]


def _ident(stem):
    s = re.sub(r"\W+", "_", stem).strip("_").lower() or "df"
    if s[0].isdigit():
        s = "df_" + s
    if s in ("pd", "np"):
        s = "df_" + s
    base, n = s, 2
    while s in NS:
        s = f"{base}_{n}"
        n += 1
    return s


def _col_ref(frame_code, label):
    return f"{frame_code}[{label!r}]"


# ---------------------------------------------------------------- table view
def _apply_view(name, query, sort, asc, search):
    df = _frame(name)
    key = (name, id(df), df.shape, query, sort, asc, search)
    hit = _VIEW_CACHE.get("key")
    if hit == key:
        return _VIEW_CACHE["value"]
    d = df
    code = name
    note = None
    if query and query.strip():
        q = query.strip()
        d = d.query(q, local_dict={k: v for k, v in NS.items() if not k.startswith("_")})
        code += f".query({q!r})"
    if search and search.strip():
        term = search.strip()
        mask = d.astype(str).apply(lambda c: c.str.contains(term, case=False, regex=False)).any(axis=1)
        d = d[mask.to_numpy()]
        code += (
            f".pipe(lambda d: d[d.astype(str).apply(lambda c: c.str.contains({term!r}, "
            f"case=False, regex=False)).any(axis=1)])"
        )
    if sort is not None:
        if sort == -1:
            d = d.sort_index(ascending=asc, kind="stable")
            code += ".sort_index()" if asc else ".sort_index(ascending=False)"
        else:
            s = d.iloc[:, sort].reset_index(drop=True)
            try:
                order = s.sort_values(ascending=asc, na_position="last", kind="stable").index.to_numpy()
            except TypeError:
                order = s.astype(str).sort_values(ascending=asc, na_position="last", kind="stable").index.to_numpy()
                note = "Column holds mixed types, so it was sorted as text."
            d = d.iloc[order]
            lab = d.columns[sort]
            code += f".sort_values({lab!r}" + (")" if asc else ", ascending=False)")
    value = (d, code, note)
    _VIEW_CACHE["key"] = key
    _VIEW_CACHE["value"] = value
    return value


def view(name, query="", sort=None, asc=True, search="", page=0, size=50):
    df = _frame(name)
    d, code, note = _apply_view(name, query, sort, asc, search)
    pages = max(1, math.ceil(len(d) / size))
    page = min(max(0, page), pages - 1)
    chunk = d.iloc[page * size:(page + 1) * size]
    out = _table(chunk)
    out.update(
        name=name,
        total=int(len(df)),
        filtered=int(len(d)),
        page=page,
        pages=pages,
        size=size,
        start=page * size,
        code=code,
        note=note,
    )
    return out


def save_view(name, query="", sort=None, asc=True, search="", newname=""):
    d, code, _ = _apply_view(name, query, sort, asc, search)
    target = newname.strip() if newname and newname.strip() else _ident(name + "_view")
    if not target.isidentifier():
        raise ValueError(f"{target!r} isn't a valid Python name. Use letters, digits and underscores.")
    NS[target] = d.copy()
    return {"name": target, "code": f"{target} = {code}.copy()", "frames": frames()}


# ---------------------------------------------------------------- info & profile
def info(name):
    df = _frame(name)
    cols = []
    n = len(df)
    for i in range(df.shape[1]):
        s = df.iloc[:, i]
        nn = int(s.notna().sum())
        try:
            nu = int(s.nunique(dropna=True))
        except TypeError:
            nu = None
        cols.append({
            "i": i,
            "label": _label(df.columns[i]),
            "dtype": str(s.dtype),
            "kind": _kind(s.dtype),
            "nonnull": nn,
            "nulls": n - nn,
            "unique": nu,
        })
    return {
        "name": name,
        "rows": n,
        "cols": int(df.shape[1]),
        "memory": int(df.memory_usage(deep=True).sum()),
        "index_type": type(df.index).__name__,
        "index_dtype": str(df.index.dtype),
        "columns": cols,
    }


def profile(name, i):
    df = _frame(name)
    s = df.iloc[:, i]
    lab = df.columns[i]
    k = _kind(s.dtype)
    ref = _col_ref(name, lab)
    out = {"label": _label(lab), "dtype": str(s.dtype), "kind": k,
           "count": int(s.notna().sum()), "nulls": int(s.isna().sum())}
    if k in ("int", "float"):
        desc = s.describe()
        out["stats"] = [[str(a), _fcell(b)] for a, b in desc.items()]
        vals = s.dropna().astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            nb = int(min(20, max(1, len(np.unique(vals)))))
            counts, edges = np.histogram(vals, bins=nb)
            out["hist"] = {"counts": counts.tolist(),
                           "edges": [_ffloat(e) for e in edges]}
        out["code"] = f"{ref}.describe()"
    elif k == "datetime":
        nn = s.dropna()
        if len(nn):
            out["stats"] = [["min", str(nn.min())], ["max", str(nn.max())],
                            ["span", str(nn.max() - nn.min())]]
            ints = nn.astype("int64").to_numpy()
            counts, edges = np.histogram(ints, bins=int(min(20, max(1, len(np.unique(ints))))))
            out["hist"] = {"counts": counts.tolist(),
                           "edges": [pd.Timestamp(int(e)).strftime("%Y-%m-%d") for e in edges]}
        out["code"] = f"{ref}.describe()"
    else:
        try:
            vc = s.value_counts(dropna=False)
        except TypeError:
            vc = s.astype(str).value_counts(dropna=False)
        top = vc.head(12)
        out["top"] = [[_fcell(a) if _fcell(a) is not None else "NaN", int(b)] for a, b in top.items()]
        out["other"] = int(vc.iloc[12:].sum()) if len(vc) > 12 else 0
        out["distinct"] = int(len(vc))
        out["code"] = f"{ref}.value_counts(dropna=False)"
    return out


def describe(name):
    df = _frame(name)
    if df.shape[1] == 0:
        return {"empty": True, "code": f"{name}.describe()"}
    d = df.describe(include="all")
    cols = []
    data = []
    for i in range(d.shape[1]):
        cols.append({"label": _label(d.columns[i]), "dtype": str(df.dtypes.iloc[i]) if i < df.shape[1] else "", "kind": "object"})
        data.append([_fcell(v) for v in d.iloc[:, i].tolist()])
    return {"columns": cols, "data": data, "index": [str(x) for x in d.index],
            "code": f"{name}.describe(include='all')"}


# ---------------------------------------------------------------- charts
def chart(name, kind="bar", x=0, y=None, agg="sum", freq="M", bins=30):
    df = _frame(name)
    xs = df.iloc[:, x]
    xl = df.columns[x]
    ys = df.iloc[:, y] if y is not None and y >= 0 else None
    yl = df.columns[y] if ys is not None else None
    note = None

    if kind == "hist":
        if not _is_numeric(xs):
            raise ValueError("A histogram needs a numeric column.")
        r = xs.dropna().value_counts(bins=int(bins), sort=False)
        labels = [f"{_ffloat(iv.left)} – {_ffloat(iv.right)}" for iv in r.index]
        return {"chart": "bar", "labels": labels, "values": [int(v) for v in r.tolist()],
                "xlabel": _label(xl), "ylabel": "count",
                "code": f"{_col_ref(name, xl)}.value_counts(bins={int(bins)}, sort=False)"}

    if kind == "scatter":
        if ys is None or not _is_numeric(xs) or not _is_numeric(ys):
            raise ValueError("A scatter plot needs two numeric columns.")
        both = pd.DataFrame({"x": xs.to_numpy(), "y": ys.to_numpy()}).dropna()
        code = f"{name}[[{xl!r}, {yl!r}]].dropna()"
        if len(both) > 3000:
            both = both.sample(3000, random_state=0)
            note = f"Showing a random sample of 3,000 of {len(xs):,} rows."
            code += ".sample(3000, random_state=0)"
        return {"chart": "scatter", "points": [[_num(a), _num(b)] for a, b in both.itertuples(index=False)],
                "xlabel": _label(xl), "ylabel": _label(yl), "code": code, "note": note}

    # bar / line: aggregate y by x
    key = xs
    kcode = _col_ref(name, xl)
    is_dt = _kind(xs.dtype) == "datetime"
    if kind == "line" and is_dt and freq and freq != "none":
        key = xs.dt.to_period(freq)
        kcode = f"{_col_ref(name, xl)}.dt.to_period({freq!r})"

    if ys is None or agg == "count":
        r = df.groupby(key, dropna=False).size() if key is not xs else xs.value_counts(dropna=False)
        ylab = "count"
        code = (f"{name}.groupby({kcode}).size()" if key is not xs
                else f"{_col_ref(name, xl)}.value_counts(dropna=False)")
    else:
        if not _is_numeric(ys):
            raise ValueError(f"Can't take the {agg} of {_label(yl)!r} because it isn't numeric. Pick a numeric column or use count.")
        r = ys.groupby(key, dropna=False).agg(agg)
        ylab = f"{agg} of {_label(yl)}"
        code = (f"{name}.groupby({kcode})[{yl!r}].{agg}()" if key is not xs
                else f"{name}.groupby({xl!r})[{yl!r}].{agg}()")

    if kind == "line":
        try:
            r = r.sort_index()
        except TypeError:
            pass
        if len(r) > 2000:
            note = f"{len(r):,} points; consider a coarser period."
    else:
        r = r.sort_values(ascending=False)
        code += ".sort_values(ascending=False)"
        if len(r) > 40:
            note = f"Showing the top 40 of {len(r):,} groups."
            r = r.head(40)
    labels = ["NaN" if (lbl is None or (isinstance(lbl, float) and math.isnan(lbl)) or lbl is pd.NaT)
              else str(lbl) for lbl in r.index]
    return {"chart": "line" if kind == "line" else "bar", "labels": labels,
            "values": [_num(v) for v in r.tolist()], "xlabel": _label(xl), "ylabel": ylab,
            "code": code, "note": note}


# ---------------------------------------------------------------- console
def _format_tb(e):
    frames_ = [f for f in traceback.extract_tb(e.__traceback__) if f.filename != ENGINE_FILE]
    lines = []
    if frames_ and not isinstance(e, SyntaxError):
        lines.append("Traceback (most recent call last):\n")
        lines.extend(traceback.format_list(frames_))
    lines.extend(traceback.format_exception_only(type(e), e))
    return "".join(lines)


def run(src):
    _VIEW_CACHE.clear()
    buf = io.StringIO()
    result = None
    error = None
    try:
        tree = ast.parse(src, filename="<console>", mode="exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, "<console>", "exec"), NS)
            if last is not None:
                result = eval(compile(last, "<console>", "eval"), NS)
    except BaseException as e:  # noqa: BLE001 - surface everything to the user
        error = _format_tb(e)
    out = {"stdout": buf.getvalue(), "error": error, "frames": frames()}
    if result is not None:
        NS["_"] = result
        if isinstance(result, pd.Series):
            t = _table(result.to_frame(), 200)
            t["shape"] = f"Series · {len(result):,} values · dtype {result.dtype}"
            t["truncated"] = len(result) > 200
            out["table"] = t
        elif isinstance(result, pd.DataFrame):
            t = _table(result, 200)
            t["shape"] = f"DataFrame · {result.shape[0]:,} rows × {result.shape[1]:,} columns"
            t["truncated"] = len(result) > 200
            out["table"] = t
        else:
            r = repr(result)
            out["repr"] = r if len(r) < 20000 else r[:20000] + "\n…"
    return out


# ---------------------------------------------------------------- files
READERS = {
    ".csv": ("read_csv", {}),
    ".tsv": ("read_csv", {"sep": "\t"}),
    ".txt": ("read_csv", {"sep": None, "engine": "python"}),
    ".json": ("read_json", {}),
    ".jsonl": ("read_json", {"lines": True}),
    ".xlsx": ("read_excel", {}),
    ".xls": ("read_excel", {}),
    ".parquet": ("read_parquet", {}),
}


def load(path, filename):
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in READERS:
        raise ValueError(f"Can't open {ext or 'that'} files. Use CSV, TSV, JSON, JSON Lines, Excel or Parquet.")
    fn, kw = READERS[ext]
    try:
        df = getattr(pd, fn)(path, **kw)
    except ValueError:
        if ext != ".json":
            raise
        df = pd.read_json(path, lines=True)
        kw = {"lines": True}
    var = _ident(stem)
    NS[var] = df
    args = ", ".join([repr(filename)] + [f"{k}={v!r}" for k, v in kw.items()])
    return {"name": var, "code": f"{var} = pd.{fn}({args})", "frames": frames()}


def export(name, fmt):
    df = _frame(name)
    keep_index = not isinstance(df.index, pd.RangeIndex)
    path = f"/tmp/{name}.{fmt}"
    if fmt == "csv":
        df.to_csv(path, index=keep_index)
        code = f"{name}.to_csv({name + '.csv'!r}, index={keep_index})"
    elif fmt == "json":
        df.to_json(path, orient="records", date_format="iso", indent=2)
        code = f"{name}.to_json({name + '.json'!r}, orient='records', date_format='iso')"
    elif fmt == "xlsx":
        df.to_excel(path, index=keep_index)
        code = f"{name}.to_excel({name + '.xlsx'!r}, index={keep_index})"
    elif fmt == "parquet":
        df.to_parquet(path)
        code = f"{name}.to_parquet({name + '.parquet'!r})"
    else:
        raise ValueError(f"Unknown export format {fmt!r}.")
    return {"path": path, "filename": os.path.basename(path), "code": code}


def sample():
    rng = np.random.default_rng(42)
    n = 600
    products = {"Notebook": 4.5, "Pen set": 12.0, "Desk lamp": 39.0,
                "Monitor stand": 59.0, "Backpack": 48.0, "Headphones": 89.0}
    names = list(products)
    prod = rng.choice(names, n, p=[0.26, 0.22, 0.14, 0.12, 0.14, 0.12])
    base = np.array([products[p] for p in prod])
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    units = rng.poisson(2.2, n) + 1
    price = (base * (1 + rng.normal(0, 0.04, n))).round(2)
    discount = rng.choice([0.0, 0.0, 0.0, 0.10, 0.15], n)
    rating = rng.choice([1, 2, 3, 4, 4, 5, 5, 5], n).astype(float)
    rating[rng.random(n) < 0.3] = np.nan
    df = pd.DataFrame({
        "order_id": [f"ORD-{10001 + i}" for i in range(n)],
        "order_date": dates,
        "region": rng.choice(["West", "Midwest", "South", "Northeast"], n, p=[0.32, 0.2, 0.3, 0.18]),
        "channel": rng.choice(["Online", "Retail", "Wholesale"], n, p=[0.55, 0.35, 0.10]),
        "product": prod,
        "units": units,
        "unit_price": price,
        "discount": discount,
    })
    df["revenue"] = (df["units"] * df["unit_price"] * (1 - df["discount"])).round(2)
    df["returned"] = rng.random(n) < 0.06
    df["rating"] = rating
    df = df.sort_values("order_date", kind="stable").reset_index(drop=True)
    df["order_id"] = [f"ORD-{10001 + i}" for i in range(n)]
    NS["orders"] = df
    return {"name": "orders", "frames": frames()}


# ---------------------------------------------------------------- dispatch
COMMANDS = {
    "frames": lambda: {"frames": frames()},
    "view": view, "save_view": save_view, "info": info, "profile": profile,
    "describe": describe, "chart": chart, "run": run, "load": load,
    "export": export, "sample": sample,
}


def dispatch(cmd, args_json):
    args = json.loads(args_json) if args_json else {}
    try:
        res = COMMANDS[cmd](**args)
        return json.dumps({"ok": True, "result": _clean(res)})
    except Exception as e:  # noqa: BLE001
        msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        return json.dumps({"ok": False, "error": msg})


VERSION = {"pandas": pd.__version__, "numpy": np.__version__}
