"""featuresのビュー(分析用のPostgres)から、学習用のデータセット・推論の入力を作る.

仕様の最上位のsourceのビューを読む。ビューの列の契約:

- 入力のビュー(仕様のsource。例: features.entries_v1): 1頭1行で、レースの前に分かる列だけ。
  racehorsekey・racekey・race_date(date)・umabanと、仕様の特徴量の列
- ラベルのビュー(例: features.labels_v1): 1頭1行の確定後の値。
  racehorsekey・finish_order・win1_num・win1_ret

読み込むたびに、ビューの定義のハッシュ(view_digest)を仕様のview_digestと照合し、
違えば止める。ハッシュは、そのビューと、依存するビュー・関数の定義(Postgresが保存している形)
から作るため、関係のないビューを足したり変えたりしても変わらない。
学習の前にpin_view_digestで今のハッシュを仕様に書き、その仕様をモデルと一緒に保存する。
"""

import dataclasses
from datetime import date
from typing import Dict, List, Sequence

import numpy as np
from sqlalchemy import Connection, Engine, text

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.infrastructure.mysql_loader import _periods, _to_date
from jra_tools.machine_learning.usecase.table_dataset import (
    DatasetRows,
    InputDataset,
    InputRows,
    TableDataset,
    assemble_dataset,
    assemble_input,
)

# 入力のビューに必要な列
ENTRY_COLUMNS = ("racehorsekey", "racekey", "race_date", "umaban")
# ラベルのビューに必要な列
LABEL_COLUMNS = ("racehorsekey", "finish_order", "win1_num", "win1_ret")
# 既定のラベルのビュー
DEFAULT_LABELS = "features.labels_v1"

# ビューと、依存するビュー・関数を、pg_dependでたどって定義を集める。
# 表(staging)は名前だけを含める。search_pathはpg_catalogにしてから流す(名前をすべて
# スキーマ付きで出すため。接続ごとのsearch_pathでハッシュが変わらないように)。
_DIGEST_SQL = """
with recursive deps(oid, kind) as (
  select cast(:view as regclass)::oid, 'r'
  union
  select d.refobjid,
         case d.refclassid when 'pg_class'::regclass then 'r' else 'f' end
  from deps
  join pg_depend d
    on (deps.kind = 'r' and d.classid = 'pg_rewrite'::regclass
        and d.objid in (select rw.oid from pg_rewrite rw where rw.ev_class = deps.oid))
    or (deps.kind = 'f' and d.classid = 'pg_proc'::regclass and d.objid = deps.oid)
  where d.refclassid in ('pg_class'::regclass, 'pg_proc'::regclass)
    and d.refobjid <> deps.oid
)
select encode(sha256(convert_to(string_agg(def, chr(10) order by name), 'UTF8')), 'hex')
from (
  select distinct
    case kind when 'f' then oid::regprocedure::text else oid::regclass::text end as name,
    case
      when kind = 'f' then pg_get_functiondef(oid)
      when (select relkind from pg_class c where c.oid = deps.oid) in ('v', 'm')
        then 'view ' || oid::regclass::text || chr(10) || pg_get_viewdef(oid)
      else 'table ' || oid::regclass::text
    end as def
  from deps
) t
"""


def view_digest(connection: Connection, view: str) -> str:
    """ビューの定義のハッシュ.

    ビューと、依存するビュー・関数の定義から作る。空白やコメントだけの違いでは変わらない
    (Postgresが保存した形から作るため)。Postgresのメジャーバージョンを上げると、
    保存した形の出し方が変わり、ハッシュが変わることがある。

    Args:
        connection (Connection): Postgresの接続
        view (str): 「スキーマ.ビュー名」

    Raises:
        ValueError: ビューがない場合

    Returns:
        str: SHA-256の16進数

    """
    _check_exists(connection, view)
    search_path = connection.execute(text("select current_setting('search_path')"))
    previous = search_path.scalar()
    connection.execute(text("select set_config('search_path', 'pg_catalog', true)"))
    try:
        return connection.execute(text(_DIGEST_SQL), {"view": view}).scalar_one()
    finally:
        connection.execute(
            text("select set_config('search_path', :p, true)"), {"p": previous}
        )


def pin_view_digest(engine: Engine, spec: FeatureSpec) -> FeatureSpec:
    """仕様のview_digestを、今のビューのハッシュにした仕様を返す(学習の前に使う).

    Args:
        engine (Engine): Postgresのエンジン
        spec (FeatureSpec): ビューから読む仕様

    Returns:
        FeatureSpec: view_digestを書いた仕様

    """
    view = _view_of(spec)
    with engine.connect() as connection:
        return dataclasses.replace(spec, view_digest=view_digest(connection, view))


def check_view(connection: Connection, spec: FeatureSpec) -> None:
    """ビューに仕様の列があり、定義が学習時と同じかを確かめる.

    Args:
        connection (Connection): Postgresの接続
        spec (FeatureSpec): ビューから読む仕様

    Raises:
        ValueError: 列が足りない、または定義が学習時と違う場合

    """
    view = _view_of(spec)
    if spec.view_digest is None:
        raise ValueError(
            f"仕様{spec.name}にview_digestがありません(pin_view_digestで書いてください)"
        )
    _check_columns(connection, view, [*ENTRY_COLUMNS, *_columns(spec)])
    digest = view_digest(connection, view)
    if digest != spec.view_digest:
        raise ValueError(
            f"ビュー{view}の定義が、仕様{spec.name}の学習時と違います"
            f"(今{digest[:12]}、学習時{spec.view_digest[:12]})。"
            "仕様を書き直すか、ビューの版を上げてください"
        )


def load_view_input(engine: Engine, spec: FeatureSpec, ymd: int) -> InputDataset:
    """開催日の推論の入力を、ビューから作る.

    Args:
        engine (Engine): Postgresのエンジン
        spec (FeatureSpec): ビューから読む仕様(view_digestがあること)
        ymd (int): 開催日(YYYYMMDD)

    Returns:
        InputDataset: 入力と、各レースのレースキー。レースはレースキーの順

    """
    day = _to_date(ymd)
    with engine.begin() as connection:
        rows = _read(connection, spec, day, day, labels=None)
    return assemble_input(_input_rows(rows, spec), spec)


def load_view_dataset(
    engine: Engine,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: str = DEFAULT_LABELS,
) -> TableDataset:
    """期間内の学習用のデータセットを、ビューから作る. 1年ずつ読み込む.

    Args:
        engine (Engine): Postgresのエンジン
        spec (FeatureSpec): ビューから読む仕様(view_digestがあること)
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)
        labels (str): ラベルのビュー

    Returns:
        TableDataset: 学習用のデータセット。レースは開催日・レースキーの順

    """
    chunks = []
    for s, e in _periods(_to_date(start), _to_date(end)):
        with engine.begin() as connection:
            chunks.append(_read(connection, spec, s, e, labels))
    rows = {key: _concat([chunk[key] for chunk in chunks]) for key in _keys(spec)}
    return assemble_dataset(_dataset_rows(rows, spec), spec)


def _read(
    connection: Connection,
    spec: FeatureSpec,
    start: date,
    end: date,
    labels: str | None,
) -> Dict[str, np.ndarray]:
    """1トランザクションで、ビューを照合してから読む.

    先にビューを共有ロックし、読み終わるまでapply_viewsに作り直されないようにする。
    """
    view = _view_of(spec)
    for name in [view, labels] if labels else [view]:
        _check_exists(connection, name)
    quote = connection.dialect.identifier_preparer.quote
    connection.exec_driver_sql(
        f"lock table {_qualified(quote, view)}"
        + (f", {_qualified(quote, labels)}" if labels else "")
        + " in access share mode"
    )
    check_view(connection, spec)
    columns = [f"e.{quote(c)}" for c in ("racehorsekey", "racekey", "umaban")]
    columns += [f"e.{quote(c)} as {quote('f:' + c)}" for c in _columns(spec)]
    join = ""
    if labels:
        _check_columns(connection, labels, LABEL_COLUMNS)
        columns += [
            "l.racehorsekey as _label",
            "l.finish_order",
            "l.win1_num",
            "l.win1_ret",
        ]
        join = f" left join {_qualified(quote, labels)} l using (racehorsekey)"
    # 名前はすべてidentifier_preparerで引用している。値はパラメーターで渡す
    result = connection.execute(
        text(
            f"select {', '.join(columns)} from {_qualified(quote, view)} e{join}"  # noqa: S608
            " where e.race_date between :start and :end"
            " order by e.race_date, e.racekey, e.umaban"
        ),
        {"start": start, "end": end},
    )
    keys = list(result.keys())
    rows = result.all()
    data = {
        key: np.array([row[i] for row in rows], dtype=object)
        for i, key in enumerate(keys)
    }
    for column in _columns(spec):
        _check_not_null(data, f"f:{column}", f"{view}.{column}が値なしの出走があります")
    if labels:
        _check_not_null(data, "_label", "成績がありません")
        _check_not_null(data, "win1_num", "払戻データがありません")
        _check_not_null(data, "win1_ret", "払戻データがありません")
    return data


def _input_rows(data: Dict[str, np.ndarray], spec: FeatureSpec) -> InputRows:
    return InputRows(
        racehorsekeys=data["racehorsekey"],
        racekeys=data["racekey"],
        nums=data["umaban"].astype(int),
        features={c: data[f"f:{c}"] for c in _columns(spec)},
    )


def _dataset_rows(data: Dict[str, np.ndarray], spec: FeatureSpec) -> DatasetRows:
    inputs = _input_rows(data, spec)
    return DatasetRows(
        **{f.name: getattr(inputs, f.name) for f in dataclasses.fields(inputs)},
        order_of_arrival=data["finish_order"],
        win1_num=data["win1_num"],
        win1_ret=data["win1_ret"],
    )


def _keys(spec: FeatureSpec) -> List[str]:
    return [
        "racehorsekey",
        "racekey",
        "umaban",
        *[f"f:{c}" for c in _columns(spec)],
        "_label",
        "finish_order",
        "win1_num",
        "win1_ret",
    ]


def _view_of(spec: FeatureSpec) -> str:
    if spec.source is None:
        raise ValueError(
            f"仕様{spec.name}にsource(ビュー)がありません。jrdb-modelから読む仕様です"
        )
    return spec.source


def _columns(spec: FeatureSpec) -> List[str]:
    """仕様が読む列(重複を除き、最初に出た順)."""
    return list(dict.fromkeys(feature.source for feature in spec.features))


def _check_exists(connection: Connection, view: str) -> None:
    if connection.execute(text("select to_regclass(:v)"), {"v": view}).scalar() is None:
        raise ValueError(f"ビューがありません: {view}")


def _check_columns(connection: Connection, view: str, required: Sequence[str]) -> None:
    _check_exists(connection, view)
    existing = set(
        connection.execute(
            text(
                "select attname from pg_attribute"
                " where attrelid = cast(:v as regclass) and attnum > 0"
                " and not attisdropped"
            ),
            {"v": view},
        ).scalars()
    )
    missing = [column for column in required if column not in existing]
    if missing:
        raise ValueError(f"ビュー{view}に列がありません: {missing}")


def _check_not_null(data: Dict[str, np.ndarray], key: str, message: str) -> None:
    missing = np.array([value is None for value in data[key]], dtype=bool)
    if missing.any():
        raise ValueError(f"{message}: {data['racehorsekey'][missing][0]}")


def _qualified(quote, name: str) -> str:
    schema, _, view = name.partition(".")
    return f"{quote(schema)}.{quote(view)}"


def _concat(values: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(values) if values else np.array([], dtype=object)
