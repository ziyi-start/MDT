"""渐进式画像抽取器

设计文档 3.3 节: 渐进式画像构建
- 用户每轮对话后，异步触发轻量级 LLM 执行信息抽取（IE）
- 将非结构化文本转化为结构化 JSON（diseases, medications, allergies）
- 以 user_id 为主键，通过 Milvus Upsert 动态更新 Patient_Profile 集合
"""
from __future__ import annotations

import json
import logging

from schema.models import PatientProfile
from llm.client import AsyncLLMClient
from llm.prompt_templates import PROFILE_EXTRACTION_PROMPT
from rag.milvus_client import MilvusManager
from rag.embedding import dummy_embed
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)


class ProfileExtractor:
    """渐进式画像构建器

    每轮对话后:
    1. 异步调用 LLM 抽取结构化信息
    2. 合并到现有画像（去重）
    3. Upsert 到 Milvus Patient_Profile 集合（持久化）
    """

    def __init__(self, llm: AsyncLLMClient, milvus: MilvusManager | None):
        self.llm = llm
        self.milvus = milvus
        self._cache: dict[str, PatientProfile] = {}  # 内存缓存

    async def extract_and_update(self, user_id: str, text: str) -> PatientProfile:
        """从对话文本中抽取画像并更新

        流程: 读取缓存/Milvus → LLM抽取 → 合并去重 → 写入缓存+Milvus
        """
        # 尝试从缓存获取现有画像
        current = self._cache.get(user_id)

        # 缓存未命中时，尝试从 Milvus 读取（持久化画像恢复）
        if current is None and self.milvus:
            current = self._load_from_milvus(user_id)

        # 仍无画像，创建空白画像
        if current is None:
            current = PatientProfile(user_id=user_id)

        # LLM 抽取结构化信息（Guided Decoding 强制输出 JSON）
        resp = await self.llm.chat(
            messages=[
                Message(role="user", content=PROFILE_EXTRACTION_PROMPT.format(text=text)),
            ],
            response_format={"type": "json_object"},
            temperature=cfg.llm.temperatures["profile_extraction"],
        )

        try:
            extracted = json.loads(resp.content or "{}")
            # 合并到现有画像（去重），兼容 LLM 返回字符串而非列表的情况
            def _ensure_list(val) -> list:
                if isinstance(val, list):
                    return [str(v) for v in val if v]
                if isinstance(val, str) and val.strip():
                    return [val.strip()]
                return []

            current.diseases = list(set(current.diseases + _ensure_list(extracted.get("diseases"))))
            current.medications = list(set(current.medications + _ensure_list(extracted.get("medications"))))
            current.allergies = list(set(current.allergies + _ensure_list(extracted.get("allergies"))))
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning(f"画像抽取解析失败: {e}")

        # 更新缓存
        self._cache[user_id] = current

        # 持久化到 Milvus
        if self.milvus:
            try:
                await self._upsert_to_milvus(current)
            except Exception as e:
                logger.warning(f"画像 Milvus 写入失败: {e}")

        logger.info(f"画像更新: user={user_id}, diseases={current.diseases}, allergies={current.allergies}")
        return current

    def _load_from_milvus(self, user_id: str) -> PatientProfile | None:
        """从 Milvus Patient_Profile 集合读取患者画像"""
        try:
            data = self.milvus.get_profile(user_id)
            if data:
                profile = PatientProfile(
                    user_id=user_id,
                    diseases=json.loads(data.get("diseases", "[]")),
                    medications=json.loads(data.get("medications", "[]")),
                    allergies=json.loads(data.get("allergies", "[]")),
                )
                self._cache[user_id] = profile
                logger.info(f"从 Milvus 恢复画像: user={user_id}")
                return profile
        except Exception as e:
            logger.warning(f"从 Milvus 读取画像失败: {e}")
        return None

    async def _upsert_to_milvus(self, profile: PatientProfile):
        """Upsert 到 Milvus Patient_Profile 集合"""
        text_repr = f"疾病:{','.join(profile.diseases)} 用药:{','.join(profile.medications)} 过敏:{','.join(profile.allergies)}"
        vec = await dummy_embed(text_repr)

        self.milvus.upsert(
            collection_name=cfg.milvus.collections.profile,
            data=[{
                "user_id": profile.user_id,
                "embedding": vec,
                "diseases": json.dumps(profile.diseases, ensure_ascii=False),
                "medications": json.dumps(profile.medications, ensure_ascii=False),
                "allergies": json.dumps(profile.allergies, ensure_ascii=False),
            }],
        )

    def get_profile(self, user_id: str) -> PatientProfile:
        """获取缓存中的画像"""
        return self._cache.get(user_id, PatientProfile(user_id=user_id))