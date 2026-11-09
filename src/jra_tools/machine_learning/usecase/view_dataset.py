"""featuresのビューから、学習用のデータセット・推論の入力・着順の確率・評価を作る手続き.

1. 仕様の固定(学習の前): 入力のビューの今のハッシュと、カテゴリの列の「コード→番号」
   (code_categoriesから)を仕様に写す。この仕様をモデルと一緒に保存する
2. 読む: 1年ずつ、1年ごとに1つのセッションで。先にビューをロックし、仕様の列があるか・
   定義のハッシュが仕様と同じかを確かめてから読む。置き場(IFrameStore)を渡せば、
   読んだ表を残し、次からはそれを使う(メタデータが今の仕様・期間と違えば止める)
3. 組み立て: usecase/frame_dataset.py(推論はDBのcode_categoriesを読まず、仕様の控えだけ)

読む先(Postgres)はIViewReader、モデルはIFinishModel・IEvaluableModelとして受け取る。
"""

import dataclasses
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from jra_tools.machine_learning.entity.category import (
    DEFAULT_CATEGORY_SOURCES,
    pick_categories,
)
from jra_tools.machine_learning.entity.dataset import (
    LABEL_ORDERS,
    InputDataset,
    TableDataset,
)
from jra_tools.machine_learning.entity.feature_spec import Encode, FeatureSpec
from jra_tools.machine_learning.entity.input_view import (
    DEFAULT_LABELS,
    ENTRY_COLUMNS,
    LABEL_COLUMNS,
    check_columns,
    check_digest,
    check_values,
    input_columns,
)
from jra_tools.machine_learning.entity.period import year_periods
from jra_tools.machine_learning.usecase.frame_dataset import (
    frame_to_dataset,
    frame_to_input,
)
from jra_tools.machine_learning.usecase.ifinish_model import (
    IEvaluableModel,
    IFinishModel,
)
from jra_tools.machine_learning.usecase.iframe_store import IFrameStore
from jra_tools.machine_learning.usecase.iview_reader import IViewReader, IViewSession

# 読む表のキーの列(race_dateは絞り込み・並べ替えにだけ使い、表には出さない)
_KEYS = ("racehorsekey", "racekey", "umaban")
# 推論の結果の、1着〜5着の確率の列(sandbox.trial_predictionsと同じ名前)
PROBABILITY_COLUMNS = (
    "first_prob",
    "second_prob",
    "third_prob",
    "fourth_prob",
    "fifth_prob",
)


def current_digest(reader: IViewReader, view: str) -> str:
    """ビュー(「スキーマ.ビュー名」)の今の定義のハッシュ.

    Raises:
        ValueError: ビューがない場合

    """
    with reader.session() as session:
        _check_exists(session, view)
        return session.digest(view)


def pin_spec(
    reader: IViewReader,
    spec: FeatureSpec,
    category_sources: Sequence[str] = DEFAULT_CATEGORY_SOURCES,
) -> FeatureSpec:
    """仕様を固定する(学習の前に使う): ビューの今のハッシュとカテゴリの控えを写した仕様を返す.

    カテゴリの列は、探す先のcode_categoriesを合わせて項目名(列名)で引き、「コード→番号」を写す。

    Raises:
        ValueError: ビューがない、カテゴリの項目がどこにもない・2か所以上にある場合

    """
    items = list(dict.fromkeys(f.source for f in spec.category_features))
    with reader.session() as session:
        _check_exists(session, spec.source)
        digest = session.digest(spec.source)
        rows = session.code_categories(category_sources, items) if items else []
    categories = pick_categories(items, rows)
    groups = tuple(
        dataclasses.replace(
            group,
            features=tuple(
                dataclasses.replace(f, categories=categories[f.source])
                if f.encode == Encode.CATEGORY
                else f
                for f in group.features
            ),
        )
        for group in spec.groups
    )
    return dataclasses.replace(spec, view_digest=digest, groups=groups)


def read_view_frame(  # noqa: PLR0913 (期間・ラベル・置き場はどれも入口の引数)
    reader: IViewReader,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str] = DEFAULT_LABELS,
    store: Optional[IFrameStore] = None,
) -> pd.DataFrame:
    """期間内の出走を、1頭1行の表で読む. 1年ずつ、1年ごとに1つのセッションで読む.

    Args:
        reader (IViewReader): ビューの読み手
        spec (FeatureSpec): 固定した仕様
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)
        labels (Optional[str]): ラベルのビュー。Noneならラベルを読まない(推論)
        store (Optional[IFrameStore]): 読んだ表の置き場。あればそれを使い、なければ読んで残す

    Returns:
        pd.DataFrame: racehorsekey・racekey・umabanと仕様の列(ラベルのビューがあれば
            finish_orderも)。開催日・レースキー・馬番の順

    Raises:
        ValueError: 仕様が固定されていない、残した表が今の仕様・期間のものでない場合

    """
    spec.check_pinned()
    if store is None:
        return _read_years(reader, spec, start, end, labels)
    name = f"{spec.name}_{start}_{end}"
    metadata = {
        "input_digest": spec.input_digest(),
        "start": str(start),
        "end": str(end),
        "labels": labels or "",
    }
    saved = store.read(name)
    if saved is not None:
        frame, saved_metadata = saved
        differs = {
            key: (saved_metadata.get(key), value)
            for key, value in metadata.items()
            if saved_metadata.get(key) != value
        }
        if differs:
            raise ValueError(
                f"残した表{name}が、今の仕様・期間のものではありません(残した値, 今の値): {differs}"
            )
        return frame
    frame = _read_years(reader, spec, start, end, labels)
    store.write(name, frame, metadata)
    return frame


def load_view_dataset(  # noqa: PLR0913 (期間・ラベル・置き場はどれも入口の引数)
    reader: IViewReader,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: str = DEFAULT_LABELS,
    store: Optional[IFrameStore] = None,
) -> TableDataset:
    """期間内の学習用のデータセットを、ビューから作る. 1年ずつ読み込む.

    Returns:
        TableDataset: 学習用のデータセット。レースは開催日・レースキーの順

    """
    frame = read_view_frame(reader, spec, start, end, labels, store)
    return frame_to_dataset(frame, spec)


def load_view_input(reader: IViewReader, spec: FeatureSpec, ymd: int) -> InputDataset:
    """開催日(YYYYMMDD)の推論の入力を、ビューから作る.

    Returns:
        InputDataset: 入力と、各レースのレースキー。レースはレースキーの順

    """
    return frame_to_input(read_view_frame(reader, spec, ymd, ymd, labels=None), spec)


def predict_finish_probabilities(
    reader: IViewReader, spec: FeatureSpec, ymd: int, model: IFinishModel
) -> List[Tuple[str, float, float, float]]:
    """開催日の出走馬ごとの1着・2着・3着の確率を、ビューの入力からモデルで出す.

    確率だけを素の値で返す(連対率・オッズなどはPostgresのビューで出す)。
    出走馬の並びは、ビューを読んだ順(開催日・レースキー・馬番)。

    Returns:
        List[Tuple[str, float, float, float]]: (racehorsekey, 1着, 2着, 3着の確率)。
            その日に出走がなければ空(モデルを呼ばない)

    """
    frame = read_view_frame(reader, spec, ymd, ymd, labels=None)
    if frame.empty:
        return []
    probabilities = _predict(spec, frame, model)
    return [
        (key, float(p1), float(p2), float(p3))
        for key, p1, p2, p3 in zip(
            probabilities["racehorsekey"],
            *(probabilities[c] for c in PROBABILITY_COLUMNS[:3]),
            strict=True,
        )
    ]


def evaluate_period(  # noqa: PLR0913 (期間・ラベル・置き場はどれも入口の引数)
    reader: IViewReader,
    spec: FeatureSpec,
    start: int,
    end: int,
    model: IEvaluableModel,
    labels: str = DEFAULT_LABELS,
    store: Optional[IFrameStore] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """評価期間を1回読み、モデルのloss・指標と、出走馬ごとの1〜5着の確率を返す.

    評価期間は締めた年(model.racesのyear_complete)に限る。DBに書くのは使う側。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            確率(racehorsekey・first_prob〜fifth_prob。ビューを読んだ順)と、
            loss・指標(output・metric・value。outputは出力の名前、合計は"total")

    Raises:
        ValueError: 評価期間に締めていない年がある、仕様にoutput_namesがない場合

    """
    with reader.session() as session:
        years = session.incomplete_years(start, end)
    if years:
        raise ValueError(f"評価期間に締めていない年があります: {years}")
    frame = read_view_frame(reader, spec, start, end, labels, store)
    dataset = frame_to_dataset(frame, spec)
    metrics = model.evaluate(dataset.x, dataset.labels, verbose=0, return_dict=True)
    return _predict(spec, frame, model), _metrics_frame(metrics, _output_names(spec))


def _read_years(
    reader: IViewReader,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str],
) -> pd.DataFrame:
    chunks = []
    for s, e in year_periods(start, end):
        with reader.session() as session:
            chunks.append(_read(session, spec, s, e, labels))
    return pd.concat(chunks, ignore_index=True)


def _read(
    session: IViewSession,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str],
) -> pd.DataFrame:
    """1つのセッションで、ビューをロックし、照合してから読む."""
    view = spec.source
    views = [view, labels] if labels else [view]
    for name in views:
        _check_exists(session, name)
    session.lock(views)
    features = input_columns(spec)
    check_columns(view, session.columns(view), [*ENTRY_COLUMNS, *features])
    check_digest(spec, view, session.digest(view))
    if labels:
        check_columns(labels, session.columns(labels), LABEL_COLUMNS)
    columns = list(dict.fromkeys([*_KEYS, *features]))
    frame = session.read(view, columns, labels, start, end)
    return check_values(frame, spec, labels=labels is not None)


def _output_names(spec: FeatureSpec) -> Tuple[str, ...]:
    if spec.output_names is None or len(spec.output_names) != LABEL_ORDERS:
        raise ValueError(
            f"仕様{spec.name}に出力の名前(1着→{LABEL_ORDERS}着)がありません"
            "(モデルの仕様を保存するときに写してください)"
        )
    return spec.output_names


def _predict(
    spec: FeatureSpec, frame: pd.DataFrame, model: IFinishModel
) -> pd.DataFrame:
    """出走馬ごとの1〜5着の確率. 並びは表(frame)の順."""
    names = _output_names(spec)
    inputs = frame_to_input(frame, spec)
    preds = model.predict(inputs.x, verbose=0)
    outputs: Sequence[Any] = (
        [preds[name] for name in names] if isinstance(preds, Mapping) else list(preds)
    )
    # 各出走馬の、入力のレースの位置と枠(馬番 - 1)
    race = pd.Index(inputs.racekeys).get_indexer(frame["racekey"])
    slot = frame["umaban"].to_numpy(dtype=int) - 1
    result = pd.DataFrame({"racehorsekey": frame["racehorsekey"].to_numpy()})
    for column, output in zip(PROBABILITY_COLUMNS, outputs, strict=True):
        result[column] = np.asarray(output)[race, slot].astype(float)
    return result


def _metrics_frame(metrics: Mapping[str, float], names: Sequence[str]) -> pd.DataFrame:
    """evaluateの結果(名前→値)を、出力・指標・値の表にする."""
    rows = []
    for key, value in metrics.items():
        output, metric = "total", key
        for name in names:
            if key.startswith(f"{name}_"):
                output, metric = name, key[len(name) + 1 :]
                break
        rows.append({"output": output, "metric": metric, "value": float(value)})
    return pd.DataFrame(rows, columns=["output", "metric", "value"])


def _check_exists(session: IViewSession, view: str) -> None:
    if not session.exists(view):
        raise ValueError(f"ビューがありません: {view}")
