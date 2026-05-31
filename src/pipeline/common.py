"""Общие зависимости модулей вычислительного конвейера"""

from __future__ import annotations

import ast
import json
import math
import re
import urllib.request
import time
import hashlib
from collections import Counter
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ..netlogo_runner import NetLogoConfig, NetLogoRunner
