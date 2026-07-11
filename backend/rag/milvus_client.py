"""Milvus 多集合管理

管理三个核心集合:
- Medical_KB: 医学知识库（文献片段）— Dense + BM25 混合检索
- Patient_Profile: 患者画像（结构化患者信息）
- Reflection_Mem: 反思记忆（归因式反思三元组）

每个集合的 schema 严格对应设计文档中的数据模型定义。
"""
from __future__ import annotations

import logging
from typing import Optional

from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker

from config import cfg
from rag.embedding import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class MilvusManager:
    """Milvus 集合管理器

    封装集合创建、Upsert、Search 操作，
    统一管理三个业务集合的生命周期。
    """

    def __init__(self, uri: str = ""):
        self.uri = uri or cfg.milvus.uri
        self._client: Optional[MilvusClient] = None

    def connect(self):
        """连接 Milvus 服务"""
        self._client = MilvusClient(uri=self.uri)
        logger.info(f"Milvus 连接成功: {self.uri}")

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self.connect()
        return self._client

    # ============================================================
    # 集合创建 - 每个集合对应设计文档中的一个数据模型
    # ============================================================

    def create_kb_collection(self, name: str = ""):
        """创建医学知识库集合（Dense + BM25 混合检索）

        字段说明:
        - doc_id: 文档唯一标识（主键）
        - embedding: Dense 向量（512 维，BGE-small 编码）
        - sparse_embedding: BM25 稀疏向量（自动生成）
        - content: 文档内容文本（BM25 的输入字段）
        - source: 文献来源
        - department: 所属科室
        - contraindications: 禁忌信息（用于硬约束过滤）
        """
        name = name or cfg.milvus.collections.kb
        if self.client.has_collection(name):
            try:
                desc = self.client.describe_collection(name)
                fields = {f["name"]: f for f in desc.get("fields", [])}
                has_sparse = "sparse_embedding" in fields
                embed_field = fields.get("embedding", {})
                embed_dim = embed_field.get("params", {}).get("dim", 0)
                if has_sparse and embed_dim == EMBEDDING_DIM:
                    logger.info(f"集合已存在: {name} (dim={embed_dim})")
                    return
                logger.warning(f"集合 {name} 维度不匹配或缺少 sparse 字段，重建中... (schema_dim={embed_dim}, expected_dim={EMBEDDING_DIM})")
                self.client.drop_collection(name)
            except Exception:
                self.client.drop_collection(name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("department", DataType.VARCHAR, max_length=128)
        schema.add_field("contraindications", DataType.VARCHAR, max_length=2048)

        # BM25 稀疏向量字段 — 由 BM25 Function 自动从 content 生成
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

        # BM25 Function: content → sparse_embedding
        bm25_function = Function(
            name="content_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_embedding"],
        )
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type=cfg.milvus.index.type,
            metric_type=cfg.milvus.index.metric_type,
            params=cfg.milvus.index.params,
        )
        index_params.add_index(
            "sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(name, schema=schema, index_params=index_params)
        logger.info(f"创建集合: {name} (Dense + BM25)")

    def create_profile_collection(self, name: str = ""):
        """创建患者画像集合

        字段说明:
        - user_id: 用户唯一标识（主键）
        - embedding: 画像向量（用于相似画像检索）
        - diseases/medications/allergies: JSON 格式的结构化信息
        """
        name = name or cfg.milvus.collections.profile
        if self.client.has_collection(name):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("diseases", DataType.VARCHAR, max_length=2048)
        schema.add_field("medications", DataType.VARCHAR, max_length=2048)
        schema.add_field("allergies", DataType.VARCHAR, max_length=2048)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type=cfg.milvus.index.type,
            metric_type=cfg.milvus.index.metric_type,
            params=cfg.milvus.index.params,
        )

        self.client.create_collection(name, schema=schema, index_params=index_params)
        logger.info(f"创建集合: {name}")

    def create_reflection_collection(self, name: str = ""):
        """创建反思记忆集合

        字段说明:
        - triple_id: 三元组唯一标识（主键）
        - embedding: 三元组向量（用于相似意图检索）
        - intent/cause/avoid_action: 归因式反思三元组字段
        """
        name = name or cfg.milvus.collections.reflection
        if self.client.has_collection(name):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("triple_id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("intent", DataType.VARCHAR, max_length=1024)
        schema.add_field("cause", DataType.VARCHAR, max_length=2048)
        schema.add_field("avoid_action", DataType.VARCHAR, max_length=2048)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type=cfg.milvus.index.type,
            metric_type=cfg.milvus.index.metric_type,
            params=cfg.milvus.index.params,
        )

        self.client.create_collection(name, schema=schema, index_params=index_params)
        logger.info(f"创建集合: {name}")

    # ============================================================
    # 通用数据操作
    # ============================================================

    def upsert(self, collection_name: str, data: list[dict]):
        """Upsert 数据（存在则更新，不存在则插入）"""
        self.client.upsert(collection_name=collection_name, data=data)

    def insert(self, collection_name: str, data: list[dict]):
        """Insert 数据（含 BM25 Function 的集合必须用 insert）"""
        self.client.insert(collection_name=collection_name, data=data)

    def _load_collection(self, collection_name: str):
        """加载集合到内存（Milvus 搜索前必须 load）"""
        try:
            self.client.load_collection(collection_name=collection_name)
        except Exception as e:
            logger.debug(f"加载集合 {collection_name}: {e}")

    def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 0,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """Dense 向量检索

        参数:
            collection_name: 集合名称
            vector: 查询向量
            limit: 返回结果数量
            filter_expr: Milvus 过滤表达式（硬约束）
            output_fields: 返回的字段列表
        """
        if limit <= 0:
            limit = cfg.retrieval.literature_search_top_k
        # 确保集合已加载到内存
        self._load_collection(collection_name)

        results = self.client.search(
            collection_name=collection_name,
            data=[vector],
            limit=limit,
            filter=filter_expr if filter_expr else None,
            output_fields=output_fields or ["content", "source", "department", "contraindications"],
            anns_field="embedding",
        )
        if results and results[0]:
            return [
                {"id": hit.get("doc_id", hit.get("id", "")), "score": hit["distance"], **hit["entity"]}
                for hit in results[0]
            ]
        return []

    def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        query_text: str,
        limit: int = 0,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """Dense + BM25 混合检索（Milvus 原生 hybrid_search）

        同时执行 Dense 向量检索和 BM25 全文检索，通过 RRF 融合排名。

        参数:
            collection_name: 集合名称
            dense_vector: Dense 查询向量（BGE 编码）
            query_text: 查询原文（BM25 使用）
            limit: 返回结果数量
            filter_expr: Milvus 过滤表达式（硬约束）
            output_fields: 返回的字段列表
        """
        if limit <= 0:
            limit = cfg.retrieval.literature_search_top_k
        self._load_collection(collection_name)
        output_fields = output_fields or ["content", "source", "department", "contraindications"]

        # Dense 检索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="embedding",
            param={"metric_type": cfg.milvus.index.metric_type, "params": cfg.milvus.index.params},
            limit=limit,
            expr=filter_expr if filter_expr else None,
        )

        # BM25 稀疏检索请求
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},
            limit=limit,
            expr=filter_expr if filter_expr else None,
        )

        # RRF 融合
        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=cfg.retrieval.rrf_k),
            limit=limit,
            output_fields=output_fields,
        )

        if results and results[0]:
            return [
                {"id": hit.get("doc_id", hit.get("id", "")), "score": hit["distance"], **hit["entity"]}
                for hit in results[0]
            ]
        return []

    def search_keyword(
        self,
        collection_name: str,
        query: str,
        limit: int = 0,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """BM25 全文检索（仅稀疏向量）

        纯 BM25 检索，用于需要精确词匹配的场景。
        """
        if limit <= 0:
            limit = cfg.retrieval.literature_search_top_k
        output_fields = output_fields or ["content", "source", "department", "contraindications"]

        try:
            sparse_req = AnnSearchRequest(
                data=[query],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=filter_expr if filter_expr else None,
            )
            self._load_collection(collection_name)
            results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=[sparse_req],
                ranker=RRFRanker(k=cfg.retrieval.rrf_k),
                limit=limit,
                output_fields=output_fields,
            )
            if results and results[0]:
                return [
                    {"id": hit.get("doc_id", hit.get("id", "")), "score": hit["distance"], **hit["entity"]}
                    for hit in results[0]
                ]
        except Exception as e:
            logger.warning(f"BM25 检索失败: {e}")

        return []

    @staticmethod
    def _sanitize_filter_value(value: str) -> str:
        """转义 Milvus filter 表达式中的特殊字符，防止注入"""
        return value.replace('"', '').replace("'", "").replace("%", "").replace("\\", "")

    def get_profile(self, user_id: str) -> dict | None:
        """从 Patient_Profile 集合读取患者画像"""
        try:
            safe_id = self._sanitize_filter_value(user_id)
            results = self.client.query(
                collection_name=cfg.milvus.collections.profile,
                filter=f'user_id == "{safe_id}"',
                output_fields=["diseases", "medications", "allergies"],
            )
            return results[0] if results else None
        except Exception as e:
            logger.warning(f"读取患者画像失败: {e}")
            return None

    def init_all_collections(self):
        """初始化所有业务集合"""
        self.create_kb_collection()
        self.create_profile_collection()
        self.create_reflection_collection()
        logger.info("所有 Milvus 集合初始化完成")