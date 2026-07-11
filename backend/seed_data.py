"""种子数据 - 向 Milvus 注入示例医学知识

使用方式:
  MDT_USE_MILVUS=true python seed_data.py

前置条件: Milvus 服务已启动 (docker-compose up -d)

向量预计算说明:
  seed_vectors.json 由 precompute_vectors.py 提前生成，
  包含 SEED_KB 中每条知识预计算的 BGE-small 向量（512 维），
  避免启动时加载 embedding 模型。
  如需重新生成：python scripts/precompute_vectors.py
"""
from __future__ import annotations

import json
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from rag.milvus_client import MilvusManager

logger = logging.getLogger(__name__)

# 示例医学知识库数据
SEED_KB = [
    {
        "doc_id": "kb_001",
        "content": "布洛芬属于非甾体抗炎药(NSAIDs)，具有解热、镇痛、抗炎作用。常见适应症包括头痛、关节痛、痛经、牙痛及痛风急性发作。常见不良反应：胃肠道不适、溃疡出血、肾功能损害。禁忌：活动性消化道溃疡、严重肾功能不全、对阿司匹林过敏者禁用。注意事项：高血压患者慎用，可能影响降压药效果；与氯吡格雷联用增加出血风险。",
        "source": "《新编药物学》第18版",
        "department": "风湿科",
        "contraindications": "胃溃疡,肾功能不全,出血风险",
    },
    {
        "doc_id": "kb_002",
        "content": "秋水仙碱是痛风急性发作的一线用药，通过抑制中性粒细胞趋化发挥抗炎作用。用法：急性期首剂1mg，1小时后0.5mg，之后每次0.5mg每日2-3次。不良反应：腹泻、恶心呕吐、骨髓抑制。注意事项：肾功能不全者需减量，CKD 3期以上禁用常规剂量。与克拉霉素、环孢素联用可致秋水仙碱中毒。",
        "source": "《痛风诊疗指南》2023版",
        "department": "风湿科",
        "contraindications": "肾功能不全,CKD3期",
    },
    {
        "doc_id": "kb_003",
        "content": "对乙酰氨基酚（扑热息痛）是安全性较高的解热镇痛药，无抗炎作用。适应症：发热、轻中度疼痛。用法：成人每次0.5-1g，每日不超过4g。与NSAIDs相比，对乙酰氨基酚不损伤胃黏膜，不影响肾功能，不增加出血风险。痛风患者如不能使用NSAIDs，对乙酰氨基酚可作为替代选择，但对急性炎症控制效果较弱。",
        "source": "《中国疼痛医学诊疗指南》",
        "department": "风湿科",
        "contraindications": "严重肝功能不全",
    },
    {
        "doc_id": "kb_004",
        "content": "高血压患者用药注意事项：1) NSAIDs类药物（布洛芬、双氯芬酸等）可引起水钠潴留，降低降压药疗效，升高血压；2) 选择止痛药时应优先考虑对乙酰氨基酚；3) 氯吡格雷是抗血小板药物，与NSAIDs联用显著增加胃肠道出血风险；4) 降压药与多种药物存在相互作用，用药前需全面评估。",
        "source": "《中国高血压防治指南》2023修订版",
        "department": "心内科",
        "contraindications": "NSAIDs,出血风险",
    },
    {
        "doc_id": "kb_005",
        "content": "胃溃疡患者用药禁忌：1) 禁用NSAIDs类药物（布洛芬、阿司匹林、双氯芬酸等），因其抑制胃黏膜前列腺素合成，加重溃疡和出血风险；2) 如必须使用抗炎镇痛药，应在PPI（质子泵抑制剂）保护下使用COX-2选择性抑制剂（塞来昔布）；3) 痛风合并胃溃疡时，优先选择秋水仙碱或对乙酰氨基酚止痛。",
        "source": "《消化性溃疡诊断与治疗共识》2022版",
        "department": "消化科",
        "contraindications": "NSAIDs,阿司匹林,胃出血",
    },
    {
        "doc_id": "kb_006",
        "content": "氯吡格雷联合阿司匹林双抗治疗的适应症：急性冠脉综合征、冠脉支架术后。双抗治疗期间出血风险显著升高，需避免联用NSAIDs类药物。双抗治疗标准疗程：ACS后12个月，之后根据出血风险评估决定是否降阶为单抗。消化道出血高危患者应联合PPI（奥美拉唑除外，因影响氯吡格雷代谢）。",
        "source": "《冠心病抗血小板治疗中国专家共识》",
        "department": "心内科",
        "contraindications": "活动性出血,NSAIDs",
    },
    {
        "doc_id": "kb_007",
        "content": "糖尿病合并慢性肾病(CKD)患者的止痛策略：1) 避免使用NSAIDs，因可加重肾损伤并影响血糖控制；2) 对乙酰氨基酚为首选，剂量需根据肾功能调整；3) 曲马多可作为二线选择，但需注意低血糖风险；4) 严格禁用含糖制剂的止痛药；5) 控制血糖和血压是保护肾功能的基础。",
        "source": "《糖尿病肾病防治专家共识》2023",
        "department": "内分泌科",
        "contraindications": "NSAIDs,肾功能不全",
    },
    {
        "doc_id": "kb_008",
        "content": "痛风急性发作治疗方案：1) 轻中度发作：秋水仙碱（首剂1mg，1h后0.5mg）或NSAIDs（布洛芬400-600mg tid）；2) 严重发作或多关节受累：可短期使用糖皮质激素（泼尼松30mg/d，7-10天递减）；3) 合并胃溃疡者：避免NSAIDs，使用秋水仙碱+PPI保护，或对乙酰氨基酚；4) 合并肾功能不全者：避免NSAIDs和常规剂量秋水仙碱，使用糖皮质激素或低剂量秋水仙碱。",
        "source": "《中国痛风诊疗指南》2023版",
        "department": "风湿科",
        "contraindications": "胃溃疡,肾功能不全",
    },
]


async def seed_data():
    """向 Milvus 注入种子数据（独立运行模式）"""
    milvus_uri = os.getenv("MDT_MILVUS_URI", cfg.milvus.uri)

    logger.info(f"连接 Milvus: {milvus_uri}")
    milvus = MilvusManager(uri=milvus_uri)
    milvus.connect()
    milvus.init_all_collections()

    await seed_milvus(milvus)


async def seed_milvus(milvus: MilvusManager):
    """向 Milvus 注入种子数据（使用预计算向量，不加载 embedding 模型）"""
    logger.info(f"注入 {len(SEED_KB)} 条医学知识...")

    vectors_path = os.path.join(os.path.dirname(__file__), "seed_vectors.json")
    if not os.path.exists(vectors_path):
        logger.error(f"预计算向量文件不存在: {vectors_path}")
        logger.error("请先运行 scripts/precompute_vectors.py 生成 seed_vectors.json")
        return

    with open(vectors_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    data = []
    for doc in SEED_KB:
        vec_data = vectors.get(doc["doc_id"])
        if vec_data is None:
            logger.error(f"缺少预计算向量: {doc['doc_id']}")
            continue
        data.append({
            "doc_id": doc["doc_id"],
            "embedding": vec_data["embedding"],
            "content": doc["content"],
            "source": doc["source"],
            "department": doc["department"],
            "contraindications": doc["contraindications"],
        })

    milvus.insert(cfg.milvus.collections.kb, data)
    logger.info(f"成功注入 {len(data)} 条知识库数据")
    logger.info("种子数据注入完成！")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_data())