from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from plspm.config import Config, MV
from plspm.mode import Mode
from plspm.plspm import Plspm
from plspm.scheme import Scheme


SCALE_ORDER = ("PQ", "BT", "CS", "BI", "RI", "WOM")
SCALE_PATTERN = re.compile(r"^(WOM|PQ|BT|CS|BI|RI)(?=\d|[_\s-]|$)", re.IGNORECASE)
DEFAULT_PATHS = (
    ("PQ", "CS"),
    ("BT", "CS"),
    ("PQ", "BI"),
    ("BT", "BI"),
    ("CS", "RI"),
    ("BI", "RI"),
    ("CS", "WOM"),
    ("BI", "WOM"),
)


class ModelValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MeasurementResults:
    loadings: pd.DataFrame
    reliability: pd.DataFrame
    validity: pd.DataFrame
    paths: pd.DataFrame
    observations: int
    dropped_rows: int
    item_columns: tuple
    converged: bool


def detect_scale(column_name):
    match = SCALE_PATTERN.match(str(column_name).strip())
    return match.group(1).upper() if match else None


def get_measurement_items(data):
    items = {scale: [] for scale in SCALE_ORDER}
    for column in data.columns:
        scale = detect_scale(column)
        if scale:
            items[scale].append(column)
    return {scale: columns for scale, columns in items.items() if columns}


def find_out_of_range_items(data):
    out_of_range = []
    for column in data.columns:
        if detect_scale(column) is None:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if not values.between(1, 5).all():
            out_of_range.append(column)
    return out_of_range


def cronbach_alpha(data):
    item_count = data.shape[1]
    if item_count < 2:
        return np.nan
    total_variance = data.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_variance) or total_variance <= 0:
        return np.nan
    item_variance = data.var(axis=0, ddof=1).sum()
    return float(item_count / (item_count - 1) * (1 - item_variance / total_variance))


def calculate_htmt(data, scale_items):
    correlations = data.corr().abs()
    results = []
    scales = list(scale_items)
    for first_index, first in enumerate(scales):
        first_pairs = correlations.loc[scale_items[first], scale_items[first]].to_numpy()
        first_values = first_pairs[np.triu_indices(len(first_pairs), k=1)]
        first_mean = first_values.mean() if first_values.size else np.nan
        for second in scales[first_index + 1 :]:
            second_pairs = correlations.loc[scale_items[second], scale_items[second]].to_numpy()
            second_values = second_pairs[np.triu_indices(len(second_pairs), k=1)]
            second_mean = second_values.mean() if second_values.size else np.nan
            heterotrait = correlations.loc[
                scale_items[first], scale_items[second]
            ].to_numpy()
            heterotrait_mean = heterotrait.mean()
            denominator = np.sqrt(first_mean * second_mean)
            htmt = (
                heterotrait_mean / denominator
                if np.isfinite(denominator) and denominator > 0
                else np.nan
            )
            results.append(
                {
                    "Yapı 1": first,
                    "Yapı 2": second,
                    "HTMT (Klasik)": htmt,
                }
            )
    return pd.DataFrame(results, columns=["Yapı 1", "Yapı 2", "HTMT (Klasik)"])


def _path_matrix(scales, selected_paths):
    edges = tuple(dict.fromkeys(tuple(edge) for edge in selected_paths))
    if not edges:
        raise ModelValidationError("En az bir gizil değişken ilişkisi seçin.")

    indegree = {scale: 0 for scale in scales}
    outgoing = {scale: [] for scale in scales}
    for edge in edges:
        if len(edge) != 2:
            raise ModelValidationError("Her ilişki bir kaynak ve bir hedef içermelidir.")
        source, target = edge
        if source not in indegree or target not in indegree:
            raise ModelValidationError("İlişkiler yalnızca tanınan yapılar arasında kurulabilir.")
        if source == target:
            raise ModelValidationError("Bir yapı kendisine bağlanamaz.")
        outgoing[source].append(target)
        indegree[target] += 1

    order = []
    remaining = set(scales)
    while remaining:
        ready = next((scale for scale in scales if scale in remaining and indegree[scale] == 0), None)
        if ready is None:
            raise ModelValidationError("Yapı ilişkileri döngü içeremez.")
        order.append(ready)
        remaining.remove(ready)
        for target in outgoing[ready]:
            indegree[target] -= 1

    path = pd.DataFrame(0, index=order, columns=order, dtype=int)
    for source, target in edges:
        path.at[target, source] = 1
    return path, edges


def estimate_measurement_model(data, selected_paths=None):
    if data.columns.has_duplicates:
        raise ModelValidationError("Veri setindeki sütun adları benzersiz olmalıdır.")

    scale_items = get_measurement_items(data)
    if not scale_items:
        raise ModelValidationError("PQ, BT, CS, BI, RI veya WOM maddesi bulunamadı.")
    insufficient = [scale for scale, items in scale_items.items() if len(items) < 2]
    if insufficient:
        raise ModelValidationError(
            "Mode A analizi için her yapıda en az iki madde gerekir: "
            + ", ".join(insufficient)
        )

    item_columns = tuple(item for items in scale_items.values() for item in items)
    numeric_data = pd.DataFrame(index=data.index)
    invalid_items = []
    for item in item_columns:
        values = pd.to_numeric(data[item], errors="coerce")
        if (data[item].notna() & values.isna()).any() or not np.isfinite(
            values.dropna().to_numpy(dtype=float)
        ).all():
            invalid_items.append(str(item))
        numeric_data[item] = values
    if invalid_items:
        raise ModelValidationError(
            "Sayısal olmayan veya sonlu olmayan yanıt içeren maddeler: "
            + ", ".join(invalid_items)
        )

    original_rows = len(numeric_data)
    numeric_data = numeric_data.dropna()
    if len(numeric_data) < 3:
        raise ModelValidationError(
            "Eksik yanıtlar çıkarıldıktan sonra analiz için en az 3 katılımcı gerekir."
        )
    constant_items = [
        item for item in item_columns if numeric_data[item].nunique() < 2
    ]
    if constant_items:
        raise ModelValidationError(
            "Varyansı olmayan maddeler analiz edilemez: " + ", ".join(constant_items)
        )

    if selected_paths is None:
        selected_paths = [
            edge
            for edge in DEFAULT_PATHS
            if edge[0] in scale_items and edge[1] in scale_items
        ]
    path, edges = _path_matrix(list(scale_items), selected_paths)

    config = Config(path, scaled=True)
    for scale in path.index:
        config.add_lv(
            scale,
            Mode.A,
            *(MV(item) for item in scale_items[scale]),
        )

    try:
        model = Plspm(
            numeric_data,
            config,
            scheme=Scheme.PATH,
            iterations=1000,
            tolerance=1e-7,
        )
    except Exception as error:
        raise ModelValidationError(
            f"PLS-SEM hesaplaması başarısız oldu: {error}"
        ) from error

    outer_model = model.outer_model()
    loading_rows = []
    reliability_rows = []
    orientation = {}
    for scale, items in scale_items.items():
        loadings = outer_model.loc[items, "loading"].astype(float)
        orientation[scale] = -1 if loadings.sum() < 0 else 1
        loadings = loadings * orientation[scale]
        loading_rows.extend(
            {"Ölçek": scale, "Madde": item, "Dış yük": float(loadings[item])}
            for item in items
        )
        oriented_loadings = loadings.to_numpy()
        loading_sum = oriented_loadings.sum()
        residual_variance = 1 - oriented_loadings**2
        composite_denominator = loading_sum**2 + residual_variance.sum()
        composite_reliability = (
            loading_sum**2 / composite_denominator
            if np.isfinite(composite_denominator) and composite_denominator > 0
            else np.nan
        )
        reliability_rows.append(
            {
                "Ölçek": scale,
                "Cronbach's Alpha": cronbach_alpha(numeric_data[items]),
                "Composite Reliability (rho_c)": composite_reliability,
                "AVE": float(np.mean(oriented_loadings**2)),
            }
        )

    path_coefficients = model.path_coefficients()
    path_rows = [
        {
            "Kaynak": source,
            "Hedef": target,
            "Yol katsayısı": float(
                path_coefficients.at[target, source]
                * orientation[source]
                * orientation[target]
            ),
        }
        for source, target in edges
    ]
    return MeasurementResults(
        loadings=pd.DataFrame(loading_rows, columns=["Ölçek", "Madde", "Dış yük"]),
        reliability=pd.DataFrame(
            reliability_rows,
            columns=[
                "Ölçek",
                "Cronbach's Alpha",
                "Composite Reliability (rho_c)",
                "AVE",
            ],
        ),
        validity=calculate_htmt(numeric_data, scale_items),
        paths=pd.DataFrame(path_rows, columns=["Kaynak", "Hedef", "Yol katsayısı"]),
        observations=len(numeric_data),
        dropped_rows=original_rows - len(numeric_data),
        item_columns=item_columns,
        converged=True,
    )
