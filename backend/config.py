"""系统配置 - 集中管理所有可配置参数

加载优先级: 环境变量 > config/custom.yaml > config/default.yaml

使用方式:
  from config import cfg
  cfg.llm.api_key
  cfg.milvus.uri
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_yaml(name: str) -> dict:
    path = _CONFIG_DIR / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---- 数据类: 类型安全的配置访问 ----

@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 2048
    temperatures: dict = field(default_factory=lambda: {
        "router": 0.1,
        "reranker": 0.1,
        "confidence_check": 0.1,
        "reflection": 0.1,
        "profile_extraction": 0.1,
        "decision_maker": 0.1,
        "consensus": 0.3,
    })


@dataclass
class MilvusCollectionsConfig:
    kb: str = "Medical_KB"
    profile: str = "Patient_Profile"
    reflection: str = "Reflection_Mem"


@dataclass
class MilvusIndexConfig:
    type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    params: dict = field(default_factory=lambda: {"nlist": 128})


@dataclass
class MilvusConfig:
    uri: str = "http://localhost:19530"
    use_milvus: bool = False
    embedding_dim: int = 512
    collections: MilvusCollectionsConfig = field(default_factory=MilvusCollectionsConfig)
    index: MilvusIndexConfig = field(default_factory=MilvusIndexConfig)


@dataclass
class RetrievalConfig:
    top_k: int = 10
    rerank_top_k: int = 5
    quick_check_top_k: int = 3
    literature_search_top_k: int = 5
    retrieval_only_top_k: int = 5
    retrieval_only_rerank_top_k: int = 3
    rrf_k: int = 60
    over_retrieval_multiplier: int = 2
    consensus_retrieval_top_k: int = 8
    consensus_rerank_top_k: int = 4


@dataclass
class RerankerConfig:
    low_threshold: float = 0.2
    llm_weight: float = 0.7
    original_weight: float = 0.3
    ngram_weight: float = 0.5
    ngram_original_weight: float = 0.5
    content_preview_length: int = 300


@dataclass
class ConfidenceConfig:
    score_gap_threshold: float = 0.15
    min_confidence: float = 0.6


@dataclass
class DecisionMakerConfig:
    quality_threshold: float = 0.5
    retrieval_only_confidence: float = 0.3


@dataclass
class ReactConfig:
    max_iterations: int = 5


@dataclass
class ReflectionConfig:
    search_threshold: float = 0.8
    max_in_memory: int = 100
    in_memory_overlap_ratio: float = 0.3
    in_memory_min_terms: int = 2


@dataclass
class ServicesConfig:
    ner_url: str = ""
    rerank_api_url: str = ""
    ner_timeout: float = 5.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    app_version: str = "1.0.0"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    decision_maker: DecisionMakerConfig = field(default_factory=DecisionMakerConfig)
    react: ReactConfig = field(default_factory=ReactConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    default_user_id: str = "default_user"


def _dict_to_dataclass(cls, data: dict):
    """递归地将 dict 映射到 dataclass 字段"""
    if not isinstance(data, dict):
        return data
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name not in data:
            continue
        value = data[f.name]
        ft = f.type
        # 处理字符串形式的类型注解
        if isinstance(ft, str):
            if ft == "dict":
                kwargs[f.name] = value
                continue
            # 尝试解析为 dataclass
            try:
                ft_obj = eval(ft)
                if hasattr(ft_obj, "__dataclass_fields__"):
                    kwargs[f.name] = _dict_to_dataclass(ft_obj, value)
                    continue
            except Exception:
                kwargs[f.name] = value
                continue
        # dict 类型直接赋值
        origin = getattr(ft, "__origin__", None)
        if origin is dict or ft is dict:
            kwargs[f.name] = value
        elif hasattr(ft, "__dataclass_fields__"):
            kwargs[f.name] = _dict_to_dataclass(ft, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config() -> AppConfig:
    """加载配置: default.yaml + custom.yaml + 环境变量"""
    default = _load_yaml("default.yaml")
    custom = _load_yaml("custom.yaml")
    merged = _deep_merge(default, custom)

    cfg = _dict_to_dataclass(AppConfig, merged)

    # 环境变量覆盖（最高优先级）
    cfg.llm.api_key = os.getenv("MDT_LLM_API_KEY", cfg.llm.api_key)
    cfg.llm.base_url = os.getenv("MDT_LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.model = os.getenv("MDT_LLM_MODEL", cfg.llm.model)
    cfg.milvus.uri = os.getenv("MDT_MILVUS_URI", cfg.milvus.uri)
    cfg.milvus.use_milvus = os.getenv("MDT_USE_MILVUS", str(cfg.milvus.use_milvus)).lower() == "true"
    cfg.services.ner_url = os.getenv("MDT_NER_SERVICE_URL", cfg.services.ner_url)
    cfg.services.rerank_api_url = os.getenv("MDT_RERANK_API_URL", cfg.services.rerank_api_url)
    cfg.server.host = os.getenv("MDT_HOST", cfg.server.host)
    cfg.server.port = int(os.getenv("MDT_PORT", str(cfg.server.port)))

    return cfg


# 全局单例
cfg = load_config()