# MDT 配置说明

## LLM API 配置

通过环境变量 `MDT_LLM_API_KEY` 设置 API 密钥（必须）。

**PARAM**           | **VALUE**
--------------------|----------------------------------
base_url (OpenAI)   | https://api.deepseek.com
base_url (Anthropic)| https://api.deepseek.com/anthropic
api_key             | 通过环境变量 `MDT_LLM_API_KEY` 设置，不要写死在代码中
model               | deepseek-v4-flash / deepseek-v4-pro / deepseek-chat / deepseek-reasoner

## 配置方式

1. 环境变量: 复制 `.env.example` 为 `.env`，填入实际值
2. YAML 配置: 修改 `config/default.yaml` 或创建 `config/custom.yaml` 覆盖默认值
3. 优先级: 环境变量 > config/custom.yaml > config/default.yaml