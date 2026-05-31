"""Калибровка параметров агентной модели"""
from __future__ import annotations


from ..pipeline.common import *
from ..pipeline.configuration import AppConfig
from ..discussion.influence_payoff import recompute_payoff_file
from ..simulation.modeling import reference_metric_summary, run_backtest


def _safe_mean(series: pd.Series | list[Any], default: float = 0.0) -> float:
    """Возвращает среднее значение числового ряда без пропусков
    
    Args:
        series: ряд значений для усреднения
        default: значение при пустом ряде
    
    Returns:
        среднее значение или резервное значение
    """
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if values.empty:
        return float(default)
    return float(values.mean())


def _calibration_objective_settings(cfg: AppConfig) -> dict[str, float | str]:
    """Возвращает настройки целевой метрики калибровки
    
    Args:
        cfg: конфигурация проекта или приложения
    
    Returns:
        словарь с режимом целевой метрики и коэффициентом L1
    """
    mode = str(cfg.get("calibration", "objective_metric", default="nrmse") or "nrmse").strip().lower()
    aliases = {
        "only_nrmse": "nrmse",
        "nrmse_only": "nrmse",
        "nrmse+l1": "nrmse_l1",
        "nrmse_l1": "nrmse_l1",
        "combined": "nrmse_l1",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"nrmse", "nrmse_l1"}:
        mode = "nrmse"
    return {
        "metric": mode,
        "nrmse_weight": float(cfg.get("calibration", "objective_nrmse_weight", default=1.0)),
        "l1_weight": float(cfg.get("calibration", "objective_l1_weight", default=25.0)),
    }


def _calibration_metric_row(cfg: AppConfig, forecasts: pd.DataFrame, metric_summary: dict[str, Any]) -> dict[str, float | str | int | None]:
    """Собирает компоненты ошибки калибровки
    
    Args:
        cfg: конфигурация проекта или приложения
        forecasts: таблица проверок модельной симуляции
        metric_summary: сводка RMSE и nRMSE
    
    Returns:
        словарь компонент ошибки и итогового objective
    """
    f = pd.DataFrame(forecasts)
    mean_nrmse = float(metric_summary.get("matrix_nrmse") or _safe_mean(f.get("nrmse", [])))
    mean_rmse = float(metric_summary.get("matrix_rmse") or _safe_mean(f.get("rmse", [])))
    mean_l1 = _safe_mean(f.get("l1", []))
    mean_cosine = _safe_mean(f.get("cosine", []), default=1.0)
    mean_delta_l1 = _safe_mean(f.get("delta_l1", []))
    mean_delta_rmse = _safe_mean(f.get("delta_rmse", []))
    mean_actual_drift_l1 = _safe_mean(f.get("actual_drift_l1", []))
    mean_predicted_drift_l1 = _safe_mean(f.get("predicted_drift_l1", []))
    mean_underreaction_l1 = _safe_mean(f.get("drift_underreaction_l1", []))
    mean_overreaction_l1 = _safe_mean(f.get("drift_overreaction_l1", []))
    settings = _calibration_objective_settings(cfg)
    if settings["metric"] == "nrmse_l1":
        objective = settings["nrmse_weight"] * mean_nrmse + settings["l1_weight"] * mean_l1
    else:
        objective = settings["nrmse_weight"] * mean_nrmse
    drift_ratio = None
    if mean_actual_drift_l1 > 1e-12:
        drift_ratio = float(mean_predicted_drift_l1 / mean_actual_drift_l1)
    return {
        "objective": float(objective),
        "mean_nrmse": mean_nrmse,
        "mean_rmse": mean_rmse,
        "mean_l1": mean_l1,
        "mean_cosine": mean_cosine,
        "mean_delta_l1": mean_delta_l1,
        "mean_delta_rmse": mean_delta_rmse,
        "mean_actual_drift_l1": mean_actual_drift_l1,
        "mean_predicted_drift_l1": mean_predicted_drift_l1,
        "mean_drift_underreaction_l1": mean_underreaction_l1,
        "mean_drift_overreaction_l1": mean_overreaction_l1,
        "mean_drift_ratio": drift_ratio,
        "objective_metric": settings["metric"],
        "objective_nrmse_weight": settings["nrmse_weight"],
        "objective_l1_weight": settings["l1_weight"],
        "actual_range": metric_summary.get("actual_range"),
        "metric_n": metric_summary.get("metric_n"),
        "metric_m": metric_summary.get("metric_m"),
        "metric_scope": settings["metric"],
        "n": int(len(f)),
    }

def detect_prolonged(events: pd.DataFrame, min_windows: int) -> bool:
    """Выявляет длительные периоды превышения диагностического порога
    
    Args:
        events: таблица дискуссионных событий
        min_windows: минимальное число шагов дискретизации для проверки
    
    Returns:
        таблица продолжительных диагностических превышений
    """
    if events.empty or "start_index" not in events:
        return False
    starts = sorted(set(int(x) for x in events["start_index"].dropna()))
    run = 1
    for prev, cur in zip(starts, starts[1:]):
        if cur == prev + 1:
            run += 1
            if run >= min_windows:
                return True
        else:
            run = 1
    return False


def calibrate_model(
    cfg: AppConfig,
    windows_path: str | Path,
    payoff_path: str | Path,
    prob_revisions: list[float],
    alphas: list[float | None],
    noises: list[float],
    horizons: list[int],
    max_windows: int | None = 20,
    progress=None,
) -> pd.DataFrame:
    """Подбирает параметры агентной модели перебором сетки
    
    Args:
        cfg: конфигурация проекта или приложения
        windows_path: путь к таблице шагов дискретизации
        payoff_path: путь к файлу выигрышей тем
        prob_revisions: набор вероятностей пересмотра стратегии
        alphas: набор значений alpha для перебора
        noises: набор значений шума для перебора
        horizons: список горизонтов моделирования в шагах дискретизации
        max_windows: максимальное число шагов дискретизации для проверки
        progress: callback для передачи статуса выполнения
    
    Returns:
        словарь лучшей конфигурации и таблицы перебора
    """
    rows = []
    total = len(prob_revisions) * len(alphas) * len(noises)
    step = 0
    for pr in prob_revisions:
        for alpha in alphas:
            for noise in noises:
                step += 1
                if progress:
                    progress(f"Калибровка {step}/{total}: prob_revision={pr}, alpha={alpha}, noise={noise}")
                result = run_backtest(cfg, windows_path, payoff_path, horizons, pr, alpha, noise, max_windows=max_windows, save_outputs=False)
                f = result["forecasts"]
                metric_summary = reference_metric_summary(f)
                metric_row = _calibration_metric_row(cfg, f, metric_summary)
                rows.append({
                    "prob_revision": pr,
                    "alpha": alpha if alpha is not None else "none",
                    "noise": noise,
                    **metric_row,
                })
    cols = [
        "prob_revision", "alpha", "noise", "objective",
        "mean_nrmse", "mean_rmse", "mean_l1", "mean_delta_l1", "mean_drift_underreaction_l1",
        "mean_actual_drift_l1", "mean_predicted_drift_l1", "mean_drift_ratio",
        "mean_cosine", "actual_range", "metric_n", "metric_m", "metric_scope", "n",
        "objective_metric", "objective_nrmse_weight", "objective_l1_weight",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["objective", "mean_nrmse", "mean_delta_l1", "mean_l1", "mean_rmse"]).reset_index(drop=True)
    out_dir = cfg.output_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "calibration_results.csv", index=False)
    if not df.empty:
        best = df.iloc[0].to_dict()
        (out_dir / "best_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return df



def calibrate_model_optuna(
    cfg: AppConfig,
    windows_path: str | Path,
    payoff_path: str | Path,
    messages_path: str | Path,
    horizons: list[int],
    n_trials: int = 30,
    max_windows: int | None = 20,
    pr_min: float = 0.005,
    pr_max: float = 0.15,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    noise_min: float = 0.0,
    noise_max: float = 0.8,
    tune_weights: bool = True,
    n_jobs: int = 1,
    progress=None,
    alpha_mode: str = "weighted",
    sentiment_mode: str = "magnitude",
    tune_sentiment_mode: bool = False,
    fixed_w_sentiment: float | None = None,
    fixed_w_influence: float | None = None,
) -> pd.DataFrame:
    """Подбирает параметры агентной модели с помощью Optuna
    
    Args:
        cfg: конфигурация проекта или приложения
        windows_path: путь к таблице шагов дискретизации
        payoff_path: путь к файлу выигрышей тем
        messages_path: путь к таблице обработанных сообщений
        horizons: список горизонтов моделирования в шагах дискретизации
        n_trials: число испытаний Optuna
        max_windows: максимальное число шагов дискретизации для проверки
        pr_min: нижняя граница перебора вероятности пересмотра
        pr_max: верхняя граница перебора вероятности пересмотра
        alpha_min: нижняя граница перебора параметра alpha
        alpha_max: верхняя граница перебора параметра alpha
        noise_min: нижняя граница перебора параметра шума
        noise_max: верхняя граница перебора параметра шума
        tune_weights: признак подбора весов функции выигрыша
        n_jobs: число параллельных задач
        progress: callback для передачи статуса выполнения
        alpha_mode: режим выбора параметра alpha
        sentiment_mode: режим преобразования тональности
        tune_sentiment_mode: признак подбора режима преобразования тональности
        fixed_w_sentiment: фиксированный вес тональности
        fixed_w_influence: фиксированный вес авторитетности
    
    Returns:
        словарь результатов оптимизации Optuna
    """
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - сообщение показывается в интерфейсе
        raise RuntimeError("Optuna is not installed. Run: pip install optuna") from exc

    out_dir = cfg.output_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_optuna_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    messages_path = Path(messages_path)
    base_payoff_path = Path(payoff_path)

    use_live_progress = progress if int(n_jobs) <= 1 else None
    alpha_mode = str(alpha_mode or "weighted").strip().lower()
    if alpha_mode not in {"weighted", "unweighted", "search"}:
        alpha_mode = "weighted"
    fixed_sentiment_mode = str(sentiment_mode or "magnitude").strip()
    if fixed_sentiment_mode not in {"magnitude", "signed_shift", "hybrid"}:
        fixed_sentiment_mode = "magnitude"

    def objective(trial):
        """Вычисляет целевую функцию Optuna для параметров агентной модели
        
        Args:
            trial: объект испытания Optuna
        
        Returns:
            значение целевой функции для одного испытания Optuna
        """
        pr = trial.suggest_float("prob_revision", float(pr_min), float(pr_max))
        noise = trial.suggest_float("noise", float(noise_min), float(noise_max))
        if alpha_mode == "search":
            trial_alpha_mode = trial.suggest_categorical("alpha_mode", ["weighted", "unweighted"])
        else:
            trial_alpha_mode = alpha_mode
            trial.set_user_attr("alpha_mode", trial_alpha_mode)
        alpha = None if trial_alpha_mode == "unweighted" else trial.suggest_float("alpha", float(alpha_min), float(alpha_max))
        if tune_sentiment_mode:
            trial_sentiment_mode = trial.suggest_categorical("sentiment_mode", ["magnitude", "signed_shift", "hybrid"])
        else:
            trial_sentiment_mode = fixed_sentiment_mode
            trial.set_user_attr("sentiment_mode", trial_sentiment_mode)

        if tune_weights:
            w_sent = trial.suggest_float("w_sentiment", 0.0, 1.0)
            w_inf = 1.0 - w_sent
            trial_payoff = tmp_dir / f"payoff_trial_{trial.number}.csv"
            recompute_payoff_file(cfg, messages_path, trial_payoff, w_sent, w_inf, trial_sentiment_mode)
            trial_payoff_path = trial_payoff
        else:
            w_sent = float(fixed_w_sentiment if fixed_w_sentiment is not None else cfg.get("payoff", "w_sentiment", default=0.5))
            w_inf = float(fixed_w_influence if fixed_w_influence is not None else cfg.get("payoff", "w_influence", default=0.5))
            trial.set_user_attr("w_sentiment", float(w_sent))
            trial_payoff_path = base_payoff_path

        result = run_backtest(
            cfg,
            windows_path,
            trial_payoff_path,
            horizons=horizons,
            prob_revision=pr,
            alpha=alpha,
            noise=noise,
            max_windows=max_windows,
            save_outputs=False,
        )
        f = result["forecasts"]
        if f.empty:
            return 1e9
        metric_summary = reference_metric_summary(f)
        metric_row = _calibration_metric_row(cfg, f, metric_summary)
        score = float(metric_row["objective"])
        for key, value in metric_row.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("w_influence", float(w_inf))
        if use_live_progress:
            use_live_progress(
                f"Optuna trial {trial.number + 1}/{n_trials}: "
                f"objective={score:.4f}, nRMSE={float(metric_row['mean_nrmse']):.4f}%, "
                f"L1={float(metric_row['mean_l1']):.4f}"
            )
        return score

    sampler = optuna.samplers.TPESampler(seed=int(cfg.get("netlogo", "random_seed", default=42)))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), n_jobs=max(1, int(n_jobs)), show_progress_bar=False)

    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = dict(t.params)
        row.update(t.user_attrs)
        row["objective"] = float(t.value)
        row["trial"] = int(t.number)
        if "w_sentiment" in row and "w_influence" not in row:
            row["w_influence"] = 1.0 - float(row["w_sentiment"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["objective", "mean_nrmse", "mean_l1", "mean_rmse"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["trial", "objective", "prob_revision", "alpha_mode", "alpha", "noise", "mean_nrmse", "mean_rmse", "mean_l1", "mean_delta_l1", "mean_drift_underreaction_l1", "mean_cosine", "actual_range", "metric_n", "metric_m", "metric_scope", "w_sentiment", "w_influence"])
    df.to_csv(out_dir / "optuna_results.csv", index=False)
    if not df.empty:
        best = df.iloc[0].to_dict()
        (out_dir / "best_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return df
