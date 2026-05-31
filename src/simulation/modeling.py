"""Сравнение тематических распределений и запуск моделирования по истории"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig, _emit_progress
from ..data_ingestion.preparation import load_windows
from ..simulation.dynamic_state import diagnose_forecasts

def load_payoff(path: str | Path, strategy_topics: list[int]) -> np.ndarray:
    """Загружает payoff тем и согласует порядок стратегий
    
    Args:
        path: путь к файлу или директории
        strategy_topics: список тем, используемых как стратегии агентной модели
    
    Returns:
        матрица или вектор выигрышей тем
    """
    df = pd.read_csv(path)
    mapping = {int(r["topic"]): float(r["payoff"]) for _, r in df.iterrows()}
    return np.array([mapping.get(int(t), 0.5) for t in strategy_topics], dtype=float)


def distribution_to_counts(props: list[float], pop_size: int) -> list[int]:
    """Преобразует доли тем в целые численности агентов
    
    Args:
        props: доли тем в тематическом распределении
        pop_size: число синтетических агентов
    
    Returns:
        целочисленные численности агентов по стратегиям
    """
    arr = np.array(props, dtype=float)
    if arr.sum() <= 0:
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / arr.sum()
    raw = np.floor(arr * pop_size).astype(int)
    missing = int(pop_size - raw.sum())
    if missing > 0:
        order = np.argsort(-(arr * pop_size - raw))
        for idx in order[:missing]:
            raw[idx] += 1
    return raw.tolist()


def _as_float_vector(value: Any) -> np.ndarray:
    """Преобразует сохранённое значение распределения в числовой вектор
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
    
    Returns:
        числовой numpy-вектор
    """
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        try:
            return value.astype(float, copy=False).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return np.asarray([], dtype=float)
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(raw)
                return _as_float_vector(parsed)
            except Exception:
                pass
    try:
        if pd.isna(value):
            return np.asarray([], dtype=float)
    except Exception:
        pass
    try:
        return np.asarray([float(value)], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _safe_actual_range(actual_values: np.ndarray, fallback: float = 1.0) -> float:
    """Возвращает безопасный размах фактических значений для nRMSE
    
    Args:
        actual_values: фактические значения для расчёта размаха
        fallback: резервное значение при ошибке или пропуске
    
    Returns:
        положительный размах для нормировки ошибки
    """
    vals = np.asarray(actual_values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback if fallback > 1e-12 else 1.0)
    spread = float(vals.max() - vals.min())
    if spread <= 1e-12:
        return float(fallback if fallback > 1e-12 else 1.0)
    return spread


def compare_distributions(real: list[float], pred: list[float], actual_range: float | None = None) -> dict[str, float]:
    """Сравнивает фактическое и смоделированное тематические распределения
    
    Args:
        real: фактическое тематическое распределение
        pred: смоделированное тематическое распределение
        actual_range: размах фактических значений для нормировки ошибки
    
    Returns:
        словарь метрик сходства тематических распределений
    """
    a = _as_float_vector(real)
    b = _as_float_vector(pred)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    if len(a) == 0:
        return {"rmse": 0.0, "nrmse": 0.0, "l1": 0.0, "cosine": 1.0}
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if actual_range is None:
        scale = _safe_actual_range(a, fallback=1.0)
    else:
        scale = float(actual_range) if float(actual_range) > 1e-12 else 1.0
    nrmse = float((rmse / scale) * 100.0)
    l1 = float(np.sum(np.abs(diff)))
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"rmse": rmse, "nrmse": nrmse, "l1": l1, "cosine": cosine}




def compare_distribution_deltas(start: list[float], real: list[float], pred: list[float]) -> dict[str, float]:
    """Сравнивает фактический и смоделированный сдвиг от начального распределения
    
    Args:
        start: тематическое распределение начального шага дискретизации
        real: фактическое тематическое распределение будущего шага дискретизации
        pred: смоделированное тематическое распределение будущего шага дискретизации
    
    Returns:
        словарь метрик ошибки динамического сдвига
    """
    s = _as_float_vector(start)
    a = _as_float_vector(real)
    b = _as_float_vector(pred)
    n = min(len(s), len(a), len(b))
    if n <= 0:
        return {
            "delta_rmse": 0.0,
            "delta_l1": 0.0,
            "actual_drift_l1": 0.0,
            "predicted_drift_l1": 0.0,
            "drift_underreaction_l1": 0.0,
            "drift_overreaction_l1": 0.0,
        }
    s = s[:n]
    a = a[:n]
    b = b[:n]
    actual_delta = a - s
    predicted_delta = b - s
    delta_diff = actual_delta - predicted_delta
    actual_drift_l1 = float(np.sum(np.abs(actual_delta)))
    predicted_drift_l1 = float(np.sum(np.abs(predicted_delta)))
    return {
        "delta_rmse": float(np.sqrt(np.mean(delta_diff ** 2))),
        "delta_l1": float(np.sum(np.abs(delta_diff))),
        "actual_drift_l1": actual_drift_l1,
        "predicted_drift_l1": predicted_drift_l1,
        "drift_underreaction_l1": float(max(0.0, actual_drift_l1 - predicted_drift_l1)),
        "drift_overreaction_l1": float(max(0.0, predicted_drift_l1 - actual_drift_l1)),
    }

def _matrix_pair_from_forecasts(forecasts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Собирает матрицы фактических и смоделированных распределений из истории проверок
    
    Args:
        forecasts: таблица результатов проверки моделирования
    
    Returns:
        пара матриц фактических и смоделированных распределений
    """
    if forecasts is None or pd.DataFrame(forecasts).empty:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    real_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    for _, row in pd.DataFrame(forecasts).iterrows():
        real = _as_float_vector(row.get("real_props"))
        pred = _as_float_vector(row.get("predicted_props"))
        n = min(len(real), len(pred))
        if n <= 0:
            continue
        real_rows.append(real[:n])
        pred_rows.append(pred[:n])
    if not real_rows:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    width = min(len(x) for x in real_rows + pred_rows)
    if width <= 0:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    Y = np.vstack([x[:width] for x in real_rows]).astype(float)
    y = np.vstack([x[:width] for x in pred_rows]).astype(float)
    return Y, y


def reference_metric_summary(forecasts: pd.DataFrame) -> dict[str, float | None]:
    """Рассчитывает сводку RMSE и nRMSE по набору проверок моделирования
    
    Args:
        forecasts: таблица результатов проверки моделирования
    
    Returns:
        словарь сводных ошибок RMSE и nRMSE
    """
    Y, y = _matrix_pair_from_forecasts(forecasts)
    if Y.size == 0 or y.size == 0:
        return {"matrix_rmse": None, "matrix_nrmse": None, "actual_min": None, "actual_max": None, "actual_range": None, "metric_n": 0, "metric_m": 0}
    rmse = float(np.sqrt(np.mean((Y - y) ** 2)))
    actual_min = float(np.nanmin(Y))
    actual_max = float(np.nanmax(Y))
    actual_range = _safe_actual_range(Y, fallback=1.0)
    nrmse = float((rmse / actual_range) * 100.0)
    return {
        "matrix_rmse": rmse,
        "matrix_nrmse": nrmse,
        "actual_min": actual_min,
        "actual_max": actual_max,
        "actual_range": actual_range,
        "metric_n": int(Y.shape[0]),
        "metric_m": int(Y.shape[1]),
    }


def apply_reference_metric_columns(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Добавляет в таблицу проверок метрики
    
    Args:
        forecasts: таблица результатов проверки моделирования
    
    Returns:
        таблица проверок с добавленными эталонными метриками
    """
    df = pd.DataFrame(forecasts).copy()
    if df.empty or "real_props" not in df.columns or "predicted_props" not in df.columns:
        return df
    summary = reference_metric_summary(df)
    actual_range = summary.get("actual_range") or 1.0
    actual_min = summary.get("actual_min")
    actual_max = summary.get("actual_max")
    rmses: list[float | None] = []
    nrmses: list[float | None] = []
    l1s: list[float | None] = []
    cosines: list[float | None] = []
    delta_rmses: list[float | None] = []
    delta_l1s: list[float | None] = []
    actual_drifts: list[float | None] = []
    predicted_drifts: list[float | None] = []
    underreactions: list[float | None] = []
    overreactions: list[float | None] = []
    has_start_props = "start_props" in df.columns
    for _, row in df.iterrows():
        real = _as_float_vector(row.get("real_props"))
        pred = _as_float_vector(row.get("predicted_props"))
        if min(len(real), len(pred)) <= 0:
            rmses.append(None); nrmses.append(None); l1s.append(None); cosines.append(None)
            delta_rmses.append(None); delta_l1s.append(None); actual_drifts.append(None); predicted_drifts.append(None); underreactions.append(None); overreactions.append(None)
            continue
        metrics = compare_distributions(real, pred, actual_range=float(actual_range))
        rmses.append(metrics["rmse"])
        nrmses.append(metrics["nrmse"])
        l1s.append(metrics["l1"])
        cosines.append(metrics["cosine"])
        if has_start_props:
            delta_metrics = compare_distribution_deltas(row.get("start_props"), real, pred)
            delta_rmses.append(delta_metrics["delta_rmse"])
            delta_l1s.append(delta_metrics["delta_l1"])
            actual_drifts.append(delta_metrics["actual_drift_l1"])
            predicted_drifts.append(delta_metrics["predicted_drift_l1"])
            underreactions.append(delta_metrics["drift_underreaction_l1"])
            overreactions.append(delta_metrics["drift_overreaction_l1"])
        else:
            delta_rmses.append(None); delta_l1s.append(None); actual_drifts.append(None); predicted_drifts.append(None); underreactions.append(None); overreactions.append(None)
    df["rmse"] = rmses
    df["nrmse"] = nrmses
    df["l1"] = l1s
    df["cosine"] = cosines
    df["delta_rmse"] = delta_rmses
    df["delta_l1"] = delta_l1s
    df["actual_drift_l1"] = actual_drifts
    df["predicted_drift_l1"] = predicted_drifts
    df["drift_underreaction_l1"] = underreactions
    df["drift_overreaction_l1"] = overreactions
    df["metric_actual_min"] = actual_min
    df["metric_actual_max"] = actual_max
    df["metric_actual_range"] = actual_range
    df["matrix_rmse"] = summary.get("matrix_rmse")
    df["matrix_nrmse"] = summary.get("matrix_nrmse")
    df["metric_scope"] = "all_steps_all_topics_reference"
    return df


def _nrmse_threshold_percent(value: Any, default: float = 35.0) -> float:
    """Приводит порог nRMSE к процентной шкале
    
    Args:
        value: исходное значение, которое требуется нормализовать или преобразовать
        default: резервное значение
    
    Returns:
        порог nRMSE в процентах
    """
    try:
        v = float(value)
    except Exception:
        return float(default)
    if 0.0 <= v <= 1.0:
        return float(v * 100.0)
    return float(v)


def run_window_modeling(
    cfg: AppConfig,
    windows_path: str | Path,
    payoff_path: str | Path,
    start_index: int,
    horizons: list[int],
    prob_revision: float,
    alpha: float | None,
    noise: float,
) -> dict[str, Any]:
    """Запускает агентное моделирование для выбранного шага дискретизации
    
    Args:
        cfg: конфигурация проекта или приложения
        windows_path: путь к таблице шагов дискретизации
        payoff_path: путь к файлу выигрышей тем
        start_index: индекс начального шага дискретизации
        horizons: список горизонтов моделирования в шагах дискретизации
        prob_revision: вероятность пересмотра стратегии агентом
        alpha: параметр баланса собственной и наблюдаемой стратегии
        noise: вероятность случайного выбора стратегии
    
    Returns:
        словарь результатов моделирования выбранного шага
    """
    windows = load_windows(windows_path)
    if not windows:
        raise ValueError("No windows available")
    start_index = int(start_index)
    if start_index < 0 or start_index >= len(windows) - 1:
        raise ValueError("start_index must have at least one future window")
    strategy_topics = [int(t) for t in windows[start_index]["strategy_topics"]]
    payoff = load_payoff(payoff_path, strategy_topics)
    runner = NetLogoRunner(NetLogoConfig(**(cfg.get("netlogo", default={}) or {})))
    pop_size = int(cfg.get("netlogo", "pop_size", default=1000))
    initial = distribution_to_counts(windows[start_index]["real_props"], pop_size)
    max_h = max(horizons)
    sim = runner.run(payoff=payoff, initial_distribution=initial, n_steps=max_h, prob_revision=prob_revision, noise=noise, alpha=alpha)
    results = []
    for h in horizons:
        future_idx = start_index + int(h)
        if future_idx >= len(windows):
            continue
        row = sim[sim["step"] == int(h)]
        if row.empty:
            continue
        prop_cols = [c for c in sim.columns if c.startswith("prop_")]
        pred = row.iloc[-1][prop_cols].astype(float).tolist()
        real = windows[future_idx]["real_props"]
        metrics = compare_distributions(real, pred)
        results.append({
            "start_index": start_index,
            "future_index": future_idx,
            "horizon": int(h),
            "start_time": windows[start_index]["time"],
            "future_time": windows[future_idx]["time"],
            "predicted_props": pred,
            "real_props": real,
            "start_props": windows[start_index]["real_props"],
            **metrics,
            "coverage": float(windows[future_idx].get("coverage", 1.0)),
            "emerging_topic_share": float(windows[future_idx].get("emerging_topic_share", 0.0)),
            "top_tickers": windows[future_idx].get("top_tickers", {}),
        })
    if results:
        results_df = apply_reference_metric_columns(pd.DataFrame(results))
        results = results_df.to_dict(orient="records")
    return {"simulation": sim.to_dict(orient="records"), "comparisons": results}


def run_backtest(
    cfg: AppConfig,
    windows_path: str | Path,
    payoff_path: str | Path,
    horizons: list[int],
    prob_revision: float,
    alpha: float | None,
    noise: float,
    max_windows: int | None = None,
    progress=None,
    save_outputs: bool = True,
    output_subdir: str = "window_modeling",
) -> dict[str, Any]:
    """Выполняет ретроспективную проверку моделирования на истории шагов дискретизации
    
    Args:
        cfg: конфигурация проекта или приложения
        windows_path: путь к таблице шагов дискретизации
        payoff_path: путь к файлу выигрышей тем
        horizons: список горизонтов моделирования в шагах дискретизации
        prob_revision: вероятность пересмотра стратегии агентом
        alpha: параметр баланса собственной и наблюдаемой стратегии
        noise: вероятность случайного выбора стратегии
        max_windows: максимальное число шагов дискретизации для проверки
        progress: callback для передачи статуса выполнения
        save_outputs: признак сохранения результатов на диск
        output_subdir: поддиректория для результатов запуска
    
    Returns:
        словарь таблиц и путей ретроспективной проверки
    """
    windows = load_windows(windows_path)
    n = len(windows)
    last_start = n - max(horizons) - 1
    if last_start < 0:
        raise ValueError("Not enough windows for selected horizons")
    starts = list(range(0, last_start + 1))
    if max_windows:
        starts = starts[:int(max_windows)]
    rows = []
    for idx, start in enumerate(starts):
        if idx % 1 == 0:
            _emit_progress(progress, f"Моделирование шагов дискретизации: {idx+1}/{len(starts)}", idx + 1, len(starts), stage="modeling")
        out = run_window_modeling(cfg, windows_path, payoff_path, start, horizons, prob_revision, alpha, noise)
        rows.extend(out["comparisons"])
    forecasts = pd.DataFrame(rows)
    if not forecasts.empty:
        forecasts = apply_reference_metric_columns(forecasts)
    diagnostics = diagnose_forecasts(cfg, forecasts)
    out_dir = cfg.output_dir / output_subdir
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(out_dir / "window_forecasts.csv", index=False)
        diagnostics["events"].to_csv(out_dir / "local_deviation_events.csv", index=False)
        (out_dir / "rebuild_decision.json").write_text(json.dumps(diagnostics["rebuild_decision"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"forecasts": forecasts, **diagnostics, "output_dir": str(out_dir)}
