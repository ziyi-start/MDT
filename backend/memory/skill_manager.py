"""技能管理器 - 从成功回答中提取可复用 Skills

设计理念（参考消息中的 Agent 自进化）:
  "Agent 可以从用户后续反馈中抽取可复用规则，
   经过 add/merge/discard 决策后沉淀，
   并记录 provenance、版本快照和 usage stats"

核心流程:
  成功回答 → LLM 提取技能 → 检索已有相似技能 → add/merge/discard 决策 → 存储

与 ReflectionManager 的职责分离:
  - ReflectionManager: 从失败中提取"不要做什么"（避坑）
  - SkillManager:    从成功中提取"应该怎么做"（最佳实践）
"""
from __future__ import annotations

import json
import uuid
import time
import logging
from datetime import datetime
from typing import Optional

from schema.models import Skill
from llm.client import AsyncLLMClient
from llm.prompt_templates import SKILL_EXTRACTION_PROMPT, SKILL_MERGE_PROMPT
from rag.milvus_client import MilvusManager
from rag.embedding import dummy_embed, extract_chinese_terms
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)

_IN_MEMORY_SKILLS: list[dict] = []


class SkillManager:
    """技能管理器 - 从成功回答中提取、存储和检索可复用技能"""

    def __init__(self, llm: AsyncLLMClient, milvus: MilvusManager | None):
        self.llm = llm
        self.milvus = milvus
        self._max_in_memory = cfg.skill.max_in_memory

    # ============================================================
    # 技能提取 - 从成功回答中提炼可复用规则
    # ============================================================

    async def extract_from_success(
        self,
        query: str,
        answer: str,
        route_path: str,
        departments: list[str],
    ) -> Optional[Skill]:
        """从一次成功的回答中提取可复用技能

        返回 None 表示该回答不可提炼出技能（质量低或过于特化）
        """
        resp = await self.llm.chat(
            messages=[Message(
                role="user",
                content=SKILL_EXTRACTION_PROMPT.format(
                    query=query,
                    answer=answer,
                    route_path=route_path,
                    departments=", ".join(departments) if departments else "无",
                ),
            )],
            response_format={"type": "json_object"},
            temperature=cfg.llm.temperatures.get("skill_extraction", 0.1),
        )

        try:
            data = json.loads(resp.content or "{}")
            intent = data.get("intent", "").strip()
            action = data.get("action", "").strip()
            extracted_depts = data.get("departments", [])

            if not intent or not action:
                logger.info("回答质量不足以提取技能，跳过")
                return None

            skill = Skill(
                intent=intent,
                action=action,
                departments=extracted_depts or departments,
                source_query=query,
                route_path=route_path,
                provenance=f"query:{datetime.now().isoformat()}",
                version=1,
                usage_count=0,
                status="active",
            )
            return skill

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"技能提取解析失败: {e}")
            return None

    # ============================================================
    # add/merge/discard 决策
    # ============================================================

    async def resolve_conflict(self, new_skill: Skill, existing: Skill) -> Skill:
        """新旧技能冲突时，LLM 做 add/merge/discard 决策

        返回: 决策后的 Skill（可能是新技能、合并技能，或空表示 discard）
        """
        resp = await self.llm.chat(
            messages=[Message(
                role="user",
                content=SKILL_MERGE_PROMPT.format(
                    old_version=existing.version,
                    old_intent=existing.intent,
                    old_action=existing.action,
                    new_intent=new_skill.intent,
                    new_action=new_skill.action,
                ),
            )],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        try:
            data = json.loads(resp.content or "{}")
            decision = data.get("decision", "add")

            if decision == "discard":
                logger.info(f"Skill 决策 discard: {data.get('reason', '')}")
                return Skill()  # 空 skill 表示丢弃
            elif decision == "merge":
                merged = Skill(
                    intent=data.get("merged_intent", existing.intent),
                    action=data.get("merged_action", existing.action),
                    departments=list(set(existing.departments + new_skill.departments)),
                    source_query=f"{existing.source_query} | {new_skill.source_query}",
                    route_path=new_skill.route_path or existing.route_path,
                    provenance=f"{existing.skill_id}+new",
                    version=existing.version + 1,
                    usage_count=existing.usage_count,
                    status="active",
                )
                logger.info(f"Skill 决策 merge (v{existing.version}→v{merged.version})")
                return merged
            else:
                logger.info(f"Skill 决策 add: 保留新旧两条")
                return new_skill
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Skill 决策解析失败，默认 add: {e}")
            return new_skill

    # ============================================================
    # 存储
    # ============================================================

    async def store(self, skill: Skill) -> bool:
        """存储技能到 Milvus 或内存"""
        if not skill.intent or not skill.action:
            return False

        # 检查相似已有技能
        existing = await self._find_similar(skill.intent, threshold=cfg.skill.similarity_threshold)
        if existing:
            resolved = await self.resolve_conflict(skill, existing)
            if not resolved.intent:
                logger.info(f"Skill {skill.intent[:30]} 被 discard")
                return False

            if resolved.version > existing.version:
                existing.status = "superseded"
                await self._save_to_storage(existing)
                skill = resolved

        skill.skill_id = str(uuid.uuid4())
        await self._save_to_storage(skill)
        logger.info(f"Skill 存储: intent={skill.intent[:40]}... v{skill.version}")
        return True

    async def _save_to_storage(self, skill: Skill):
        """持久化存储"""
        text_repr = f"{skill.intent} {skill.action}"
        vec = await dummy_embed(text_repr)

        data = {
            "skill_id": skill.skill_id,
            "embedding": vec,
            "intent": skill.intent,
            "action": skill.action,
            "departments": json.dumps(skill.departments, ensure_ascii=False),
            "source_query": skill.source_query,
            "route_path": skill.route_path,
            "provenance": skill.provenance,
            "version": skill.version,
            "usage_count": skill.usage_count,
            "last_used": skill.last_used,
            "status": skill.status,
        }

        if self.milvus:
            try:
                self.milvus.upsert(
                    collection_name=cfg.milvus.collections.skill,
                    data=[data],
                )
            except Exception as e:
                logger.warning(f"Skill Milvus 写入失败: {e}")
                self._store_in_memory(data)
        else:
            self._store_in_memory(data)

    def _store_in_memory(self, data: dict):
        global _IN_MEMORY_SKILLS
        if len(_IN_MEMORY_SKILLS) >= self._max_in_memory:
            _IN_MEMORY_SKILLS = _IN_MEMORY_SKILLS[-self._max_in_memory // 2:]
        _IN_MEMORY_SKILLS.append(data)

    # ============================================================
    # 检索
    # ============================================================

    async def search_skills(self, query: str, threshold: float = 0.0) -> list[Skill]:
        """检索与当前查询相关的技能

        返回: 按相关性排序的活跃技能列表
        """
        if threshold <= 0:
            threshold = cfg.skill.search_threshold

        skills = []

        if self.milvus:
            try:
                vec = await dummy_embed(query)
                results = self.milvus.search(
                    collection_name=cfg.milvus.collections.skill,
                    vector=vec,
                    limit=cfg.skill.max_return,
                    filter_expr='status == "active"',
                    output_fields=["skill_id", "intent", "action", "departments",
                                   "source_query", "version", "usage_count", "last_used", "status"],
                )
                for r in results:
                    if r.get("score", 0) >= threshold:
                        skill = Skill(
                            skill_id=r.get("skill_id", ""),
                            intent=r.get("intent", ""),
                            action=r.get("action", ""),
                            departments=json.loads(r.get("departments", "[]")),
                            source_query=r.get("source_query", ""),
                            version=r.get("version", 1),
                            usage_count=r.get("usage_count", 0),
                            last_used=r.get("last_used", ""),
                            status=r.get("status", "active"),
                        )
                        skills.append(skill)

                        await self._increment_usage(skill.skill_id)

            except Exception as e:
                logger.warning(f"Skill Milvus 检索失败: {e}")

        # 内存模式
        if not skills and _IN_MEMORY_SKILLS:
            query_terms = extract_chinese_terms(query)
            scored = []
            for s in _IN_MEMORY_SKILLS:
                if s.get("status", "active") != "active":
                    continue
                intent_terms = extract_chinese_terms(s.get("intent", ""))
                overlap = len(query_terms & intent_terms)
                ratio = overlap / max(len(query_terms), 1) if query_terms else 0
                if ratio >= threshold:
                    scored.append((ratio, s))
            scored.sort(key=lambda x: x[0], reverse=True)

            for score, s in scored[:cfg.skill.max_return]:
                skills.append(Skill(
                    skill_id=s.get("skill_id", ""),
                    intent=s.get("intent", ""),
                    action=s.get("action", ""),
                    departments=json.loads(s.get("departments", "[]")),
                    source_query=s.get("source_query", ""),
                    version=s.get("version", 1),
                    usage_count=s.get("usage_count", 0) + 1,
                    status=s.get("status", "active"),
                ))

        return skills

    async def _increment_usage(self, skill_id: str):
        """技能被检索到时，递增使用计数"""
        if not self.milvus or not skill_id:
            return
        try:
            results = self.milvus.client.query(
                collection_name=cfg.milvus.collections.skill,
                filter=f'skill_id == "{skill_id}"',
                output_fields=["usage_count"],
            )
            if results:
                current = results[0].get("usage_count", 0)
                self.milvus.client.upsert(
                    collection_name=cfg.milvus.collections.skill,
                    data=[{
                        "skill_id": skill_id,
                        "usage_count": current + 1,
                        "last_used": datetime.now().isoformat(),
                    }],
                )
        except Exception as e:
            logger.debug(f"Skill usage 更新失败: {e}")

    # ============================================================
    # 相似度查找
    # ============================================================

    async def _find_similar(self, intent: str, threshold: float = 0.0) -> Optional[Skill]:
        """查找与给定 intent 相似的已有技能"""
        if threshold <= 0:
            threshold = cfg.skill.similarity_threshold

        if self.milvus:
            try:
                vec = await dummy_embed(intent)
                results = self.milvus.search(
                    collection_name=cfg.milvus.collections.skill,
                    vector=vec,
                    limit=1,
                    filter_expr='status == "active"',
                    output_fields=["skill_id", "intent", "action", "departments",
                                   "source_query", "version", "usage_count", "status"],
                )
                if results and results[0].get("score", 0) >= threshold:
                    r = results[0]
                    return Skill(
                        skill_id=r.get("skill_id", ""),
                        intent=r.get("intent", ""),
                        action=r.get("action", ""),
                        departments=json.loads(r.get("departments", "[]")),
                        source_query=r.get("source_query", ""),
                        version=r.get("version", 1),
                        usage_count=r.get("usage_count", 0),
                        status=r.get("status", "active"),
                    )
            except Exception as e:
                logger.warning(f"Skill 相似查找失败: {e}")

        return None

    # ============================================================
    # 统计
    # ============================================================

    def stats(self) -> dict:
        return {
            "total_in_memory": len(_IN_MEMORY_SKILLS),
            "milvus_available": self.milvus is not None,
        }
