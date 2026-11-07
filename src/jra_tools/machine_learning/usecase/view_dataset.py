"""featuresのビューから、学習用のデータセット・推論の入力・着順の確率を作る手続き.

仕様の最上位のsourceのビューを読む(列の契約はentity/input_view.py)。
読み込みは1年ずつ、1年ごとに1つのセッションで行う。セッションの中で、先にビューをロックし、
仕様の列があるか・定義のハッシュが学習時(仕様のview_digest)と同じかを確かめてから読む。
学習の前にpin_view_digestで今のハッシュを仕様に書き、その仕様をモデルと一緒に保存する。

読む先(Postgres)はIViewReader、モデルはIFinishModelとして受け取る。
テンソルへの組み立ては、usecase/frame_dataset.pyが行う。
"""

import dataclasses
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.entity.input_view import (
    DEFAULT_LABELS,
    ENTRY_COLUMNS,
    LABEL_COLUMNS,
    check_columns,
    check_digest,
    check_values,
    input_columns,
    require_view_digest,
    source_view,
)
from jra_tools.machine_learning.entity.period import year_periods
from jra_tools.machine_learning.usecase.frame_dataset import (
    InputDataset,
    frame_to_dataset,
    frame_to_input,
)
from jra_tools.machine_learning.usecase.ifinish_model import (
    PROBABILITY_OUTPUTS,
    IFinishModel,
)
from jra_tools.machine_learning.usecase.iview_reader import IViewReader, IViewSession
from jra_tools.machine_learning.usecase.table_dataset import TableDataset

# 読む表のキーの列(race_dateは絞り込み・並べ替えにだけ使い、表には出さない)
_KEYS = ("racehorsekey", "racekey", "umaban")


def pin_view_digest(reader: IViewReader, spec: FeatureSpec) -> FeatureSpec:
    """仕様のview_digestを、今のビューのハッシュにした仕様を返す(学習の前に使う).

    Raises:
        ValueError: 仕様にsourceがない、またはビューがない場合

    """
    view = source_view(spec)
    with reader.session() as session:
        _check_exists(session, view)
        return dataclasses.replace(spec, view_digest=session.digest(view))


def check_view(session: IViewSession, spec: FeatureSpec) -> None:
    """ビューに仕様の列があり、定義が学習時と同じかを確かめる.

    Raises:
        ValueError: view_digestがない、ビュー・列がない、または定義が学習時と違う場合

    """
    view = source_view(spec)
    require_view_digest(spec)
    _check_exists(session, view)
    check_columns(view, session.columns(view), [*ENTRY_COLUMNS, *input_columns(spec)])
    check_digest(spec, view, session.digest(view))


def read_view_frame(
    reader: IViewReader,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    """期間内の出走を、1頭1行の表で読む. 1年ずつ、1年ごとに1つのセッションで読む.

    Args:
        reader (IViewReader): ビューの読み手
        spec (FeatureSpec): ビューから読む仕様(view_digestがあること)
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)
        labels (Optional[str]): ラベルのビュー。Noneならラベル・払戻を読まない(推論)

    Returns:
        pd.DataFrame: racehorsekey・racekey・umabanと仕様の列(ラベルのビューがあれば
            finish_order・win1_num・win1_retも)。開催日・レースキー・馬番の順

    """
    chunks = []
    for s, e in year_periods(start, end):
        with reader.session() as session:
            chunks.append(_read(session, spec, s, e, labels))
    return pd.concat(chunks, ignore_index=True)


def load_view_dataset(
    reader: IViewReader,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: str = DEFAULT_LABELS,
) -> TableDataset:
    """期間内の学習用のデータセットを、ビューから作る. 1年ずつ読み込む.

    Returns:
        TableDataset: 学習用のデータセット。レースは開催日・レースキーの順

    """
    return frame_to_dataset(read_view_frame(reader, spec, start, end, labels), spec)


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
    inputs = frame_to_input(frame, spec)
    preds = model.predict(inputs.x, verbose=0)
    # 各出走馬の、入力のレースの位置と枠(馬番 - 1)
    race = pd.Index(inputs.racekeys).get_indexer(frame["racekey"])
    slot = frame["umaban"].to_numpy(dtype=int) - 1
    probs = [np.asarray(preds[name])[race, slot] for name in PROBABILITY_OUTPUTS]
    return [
        (key, float(p1), float(p2), float(p3))
        for key, p1, p2, p3 in zip(frame["racehorsekey"], *probs, strict=True)
    ]


def _read(
    session: IViewSession,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str],
) -> pd.DataFrame:
    """1つのセッションで、ビューをロックし、照合してから読む."""
    view = source_view(spec)
    views = [view, labels] if labels else [view]
    for name in views:
        _check_exists(session, name)
    session.lock(views)
    check_view(session, spec)
    features = input_columns(spec)
    if labels:
        check_columns(labels, session.columns(labels), LABEL_COLUMNS)
    columns = list(dict.fromkeys([*_KEYS, *features]))
    frame = session.read(view, columns, labels, start, end)
    return check_values(frame, view, features, labels=labels is not None)


def _check_exists(session: IViewSession, view: str) -> None:
    if not session.exists(view):
        raise ValueError(f"ビューがありません: {view}")
