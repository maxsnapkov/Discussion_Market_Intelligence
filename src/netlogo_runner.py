"""Запуск агентной модели NetLogo из Python-конвейера

Модуль формирует входные файлы, создаёт временный эксперимент BehaviorSpace
вызывает NetLogo в headless-режиме и возвращает траекторию долей стратегий
"""

from __future__ import annotations

"""Модуль не реализует эволюционную игру на стороне Python

Python здесь только записывает входные CSV-файлы, встраивает эксперимент BehaviorSpace
во временную NetLogo-модель, запускает headless-режим и читает итоговую таблицу
"""


import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


class NetLogoError(RuntimeError):
    """Исключение запуска агентной модели NetLogo
    
    Используется, когда модель не удалось выполнить или прочитать результаты симуляции
    """
    pass


@dataclass
class NetLogoConfig:
    """Параметры интеграции с NetLogo
    
    Dataclass задаёт путь к исполняемому файлу NetLogo, модели и временной директории
    
    Attributes:
        executable: путь к NetLogo Console или значение None для автоматического поиска
        model_path: путь к файлу модели относительно корня проекта или абсолютный путь
        work_dir: директория для временных файлов запуска
    """
    executable: Optional[str] = None
    model_path: str = "netlogo/evolutionary_discussion.nlogox"
    timeout: int = 180
    random_seed: int = 42
    pop_size: int = 1000
    keep_files: bool = False


class NetLogoRunner:
    """Обёртка над headless-запуском NetLogo-модели
    
    Класс подготавливает payoff, начальное распределение агентов и XML-эксперимент
    затем запускает модель и возвращает таблицу траектории стратегий
    
    Attributes:
        config: параметры запуска NetLogo
        repo_root: корень репозитория для разрешения относительных путей
    """
    def __init__(self, config: NetLogoConfig, repo_root: Path | None = None):
        """Инициализирует объект запуска NetLogo-модели
        
        Args:
            config: параметры запуска или конфигурационный объект
            repo_root: корень репозитория для разрешения относительных путей
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        self.config = config
        self.repo_root = repo_root or Path.cwd()
        self.model_path = self._resolve_path(config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"NetLogo model not found: {self.model_path}")

    def run(
        self,
        payoff: Sequence[float] | Sequence[Sequence[float]],
        initial_distribution: Sequence[int | float],
        n_steps: int,
        prob_revision: float,
        noise: float,
        alpha: float | None,
    ) -> pd.DataFrame:
        """Запускает NetLogo-модель и возвращает траекторию распределения стратегий
        
        Args:
            payoff: матрица выигрышей стратегий агентной модели
            initial_distribution: начальное распределение агентов по стратегиям
            n_steps: число шагов NetLogo-симуляции
            prob_revision: вероятность пересмотра стратегии агентом
            noise: вероятность случайного выбора стратегии
            alpha: параметр баланса собственной и наблюдаемой стратегии
        
        Returns:
            таблица pandas с траекторией долей стратегий по шагам симуляции
        """
        n_steps = int(n_steps)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        initial = [int(round(x)) for x in initial_distribution]
        if not initial or sum(initial) <= 0:
            raise ValueError("initial_distribution is empty")

        payoff_arr = np.asarray(payoff, dtype=float)
        weighted_mode = payoff_arr.ndim == 1
        if weighted_mode and payoff_arr.size != len(initial):
            raise ValueError("payoff vector length must match strategy count")
        if not weighted_mode:
            rows, cols = payoff_arr.shape
            if rows != cols or rows != len(initial):
                raise ValueError("payoff matrix must be square and match strategy count")

        executable = self._resolve_executable()
        temp_root = Path(tempfile.mkdtemp(prefix="vkr_netlogo_"))
        try:
            payoff_file = temp_root / "payoff.csv"
            initial_file = temp_root / "initial_distribution.csv"
            output_file = temp_root / "netlogo_history.csv"
            table_file = temp_root / "behaviorspace_table.csv"
            experiment_file = temp_root / "experiment.xml"
            runtime_model = temp_root / self.model_path.name

            self._write_payoff(payoff_file, payoff_arr)
            self._write_initial(initial_file, initial)
            self._write_experiment(
                experiment_file=experiment_file,
                payoff_file=payoff_file,
                initial_file=initial_file,
                output_file=output_file,
                n_steps=n_steps,
                prob_revision=prob_revision,
                noise=noise,
                alpha=alpha,
                weighted_mode=weighted_mode,
            )
            self._write_runtime_model(runtime_model, experiment_file)

            command = [
                executable,
                "--headless",
                "--model", str(runtime_model),
                "--experiment", "python-run",
                "--table", str(table_file),
                "--threads", "1",
            ]
            completed = subprocess.run(
                command,
                cwd=str(self.repo_root),
                text=True,
                capture_output=True,
                timeout=int(self.config.timeout),
            )
            if completed.returncode != 0:
                raise NetLogoError(
                    "NetLogo headless run failed.\n"
                    f"Command: {' '.join(command)}\n"
                    f"STDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )
            if not output_file.exists():
                table_text = table_file.read_text(encoding="utf-8", errors="ignore") if table_file.exists() else ""
                raise NetLogoError(
                    "NetLogo finished but did not create output file.\n"
                    f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}\n"
                    f"BehaviorSpace table:\n{table_text}"
                )
            return pd.read_csv(output_file)
        finally:
            if not self.config.keep_files:
                shutil.rmtree(temp_root, ignore_errors=True)

    def _resolve_path(self, path: str) -> Path:
        """Разрешает относительный путь относительно корня репозитория
        
        Args:
            path: путь к файлу или директории
        
        Returns:
            абсолютный путь к файлу или директории
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return self.repo_root / p

    def _resolve_executable(self) -> str:
        """Находит исполняемый файл NetLogo Console
        
        Returns:
            путь к найденному исполняемому файлу NetLogo Console
        """
        if self.config.executable:
            p = Path(self.config.executable)
            if p.exists():
                return str(p)
        for env in ["NETLOGO_CONSOLE", "NETLOGO_HEADLESS"]:
            value = os.getenv(env)
            if value and Path(value).exists():
                return value
        home = os.getenv("NETLOGO_HOME")
        if home:
            candidates = [
                Path(home) / "NetLogo_Console.exe",
                Path(home) / "netlogo-headless.bat",
                Path(home) / "NetLogo_Console",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)
        common = [
            Path("C:/Program Files/NetLogo 7.0.4/NetLogo_Console.exe"),
            Path("C:/Program Files/NetLogo 7.0.3/NetLogo_Console.exe"),
            Path("C:/Program Files/NetLogo 6.4.0/netlogo-headless.bat"),
        ]
        for c in common:
            if c.exists():
                return str(c)
        raise NetLogoError(
            "NetLogo executable not found. Set NETLOGO_HOME or NETLOGO_CONSOLE, "
            "or specify netlogo.executable in config/app_config.yaml."
        )

    @staticmethod
    def _write_payoff(path: Path, payoff: np.ndarray) -> None:
        """Записывает матрицу выигрышей во временный файл
        
        Args:
            path: путь к файлу или директории
            payoff: матрица выигрышей стратегий агентной модели
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if payoff.ndim == 1:
                for value in payoff:
                    writer.writerow([float(value)])
            else:
                for row in payoff:
                    writer.writerow([float(x) for x in row])

    @staticmethod
    def _write_initial(path: Path, initial: Sequence[int]) -> None:
        """Записывает начальное распределение агентов во временный файл
        
        Args:
            path: путь к файлу или директории
            initial: начальное распределение агентов по стратегиям
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(x) for x in initial])

    @staticmethod
    def _nl_path(path: Path) -> str:
        """Преобразует путь в формат, безопасный для NetLogo
        
        Args:
            path: путь к файлу или директории
        
        Returns:
            преобразованное значение: путь в формат, безопасный для NetLogo
        """
        return path.resolve().as_posix()

    @staticmethod
    def _xml_string(value: str) -> str:
        """Экранирует строку для XML-файла эксперимента
        
        Args:
            value: исходное значение, которое требуется нормализовать или преобразовать
        
        Returns:
            экранированная строка: строку для XML-файла эксперимента
        """
        return "&quot;" + escape(value) + "&quot;"

    def _write_experiment(
        self,
        experiment_file: Path,
        payoff_file: Path,
        initial_file: Path,
        output_file: Path,
        n_steps: int,
        prob_revision: float,
        noise: float,
        alpha: float | None,
        weighted_mode: bool,
    ) -> None:
        """Создаёт XML-файл BehaviorSpace для headless-запуска NetLogo
        
        Args:
            experiment_file: путь к XML-файлу эксперимента BehaviorSpace
            payoff_file: путь к файлу payoff для NetLogo
            initial_file: путь к файлу начального распределения агентов
            output_file: путь к выходному файлу
            n_steps: число шагов NetLogo-симуляции
            prob_revision: вероятность пересмотра стратегии агентом
            noise: вероятность случайного выбора стратегии
            alpha: параметр баланса собственной и наблюдаемой стратегии
            weighted_mode: признак использования взвешенной матрицы выигрышей
        
        Returns:
            None: функция изменяет состояние, файл или интерфейс без отдельного возвращаемого значения
        """
        values = {
            "payoff-file": self._xml_string(self._nl_path(payoff_file)),
            "initial-distribution-file": self._xml_string(self._nl_path(initial_file)),
            "output-file": self._xml_string(self._nl_path(output_file)),
            "prob-revision": str(float(prob_revision)),
            "noise": str(float(noise)),
            "alpha": str(float(alpha if alpha is not None else 0.0)),
            "weighted-mode?": "true" if weighted_mode else "false",
            "n-steps": str(int(n_steps)),
            "random-seed-value": str(int(self.config.random_seed)),
        }
        variables = []
        for name, value in values.items():
            variables.append(
                f'      <enumeratedValueSet variable="{escape(name)}">\n'
                f'        <value value="{value}"></value>\n'
                f'      </enumeratedValueSet>'
            )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<experiments>\n'
            f'  <experiment name="python-run" repetitions="1" sequentialRunOrder="true" '
            f'runMetricsEveryStep="false" timeLimit="{int(n_steps)}">\n'
            '    <setup>setup</setup>\n'
            '    <go>go</go>\n'
            '    <metrics>\n'
            '      <metric>count players</metric>\n'
            '    </metrics>\n'
            '    <constants>\n'
            + "\n".join(variables) +
            '\n    </constants>\n'
            '  </experiment>\n'
            '</experiments>\n'
        )
        experiment_file.write_text(xml, encoding="utf-8")

    @staticmethod
    def _strip_xml_declaration(xml_text: str) -> str:
        """Удаляет XML-декларацию из текста модели перед встраиванием
        
        Args:
            xml_text: текст XML или NLOGOX-модели
        
        Returns:
            None: текста модели без XML-декларации
        """
        cleaned = xml_text.lstrip("\ufeff\n\r\t ")
        if cleaned.startswith("<?xml"):
            end = cleaned.find("?>")
            if end != -1:
                cleaned = cleaned[end + 2 :]
        return cleaned.strip() + "\n"

    def _write_runtime_model(self, runtime_model: Path, experiment_file: Path) -> None:
        """Создаёт временную NetLogo-модель со встроенным экспериментом
        
        Args:
            runtime_model: путь к временной модели NetLogo со встроенным экспериментом
            experiment_file: путь к XML-файлу эксперимента BehaviorSpace
        
        Returns:
            None
        """
        text = self.model_path.read_text(encoding="utf-8-sig")
        exp = self._strip_xml_declaration(experiment_file.read_text(encoding="utf-8-sig"))

        if "<experiments>" in text and "</experiments>" in text:
            before = text.split("<experiments>", 1)[0]
            after = text.split("</experiments>", 1)[1]
            text = before + exp + after
        elif text.rstrip().endswith("</model>"):
            text = text.replace("</model>", exp + "\n</model>")
        else:
            # резервная ветка для старого формата nlogo
            # она не нужна для NetLogo 7 nlogox, но помогает ручной диагностике
            text = text.rstrip() + "\n" + exp

        if text.lstrip().startswith("<?xml"):
            text = text.lstrip("\ufeff\n\r\t ")
        runtime_model.write_text(text, encoding="utf-8")
