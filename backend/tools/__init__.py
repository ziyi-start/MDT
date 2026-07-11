"""Tools 包 - 注册所有工具

导入此包即触发工具注册到 global_tool_registry。
"""
from .literature_search import literature_search, set_retriever, set_current_profile
from .drug_interaction import check_drug_interaction