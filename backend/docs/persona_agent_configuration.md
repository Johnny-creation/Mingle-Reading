# 角色 Agent 配置

Muse Reading 当前提供了完整的文学领读 Agent 运行时路径。三个已命名的 Agent 为：

- `lu-xun`
- `mark-twain`
- `zhang-ailing`

此外还有一个 `neutral` 阅读器用于非角色输出。

## 端到端已打通的内容

每个角色 Agent 现在具备直接使用所需的全部四层：

1. `Agent 配置`
   定义在 [backend/llm_memory/persona/persona_service.py](/C:/Users/21358/Desktop/MuseReading/backend/llm_memory/persona/persona_service.py) 中，包含显示名称、环境变量名和 prompt 特征。
2. `角色 RAG 知识库`
   从 `backend/assets/data/processed/personas/persona_kb/<persona_id>/` 加载。
3. `Prompt 组装`
   系统 prompt、角色证据和阅读器可见书籍上下文在生成前统一组装。
4. `真实模型调用`
   [backend/llm_memory/persona/model_client.py](/C:/Users/21358/Desktop/MuseReading/backend/llm_memory/persona/model_client.py) 调用 OpenAI 兼容的 `/v1/chat/completions` 端点。

## 必需环境变量

将 [`.env.example`](/C:/Users/21358/Desktop/MuseReading/.env.example) 复制为 `.env` 并填入你自己的本地值。后端在启动时会自动加载此根目录下的 `.env` 文件。

预期变量：

- `LU_XUN_API_KEY`
- `LU_XUN_BASE_URL`
- `LU_XUN_MODEL_NAME`
- `MARK_TWAIN_API_KEY`
- `MARK_TWAIN_BASE_URL`
- `MARK_TWAIN_MODEL_NAME`
- `ZHANG_AILING_API_KEY`
- `ZHANG_AILING_BASE_URL`
- `ZHANG_AILING_MODEL_NAME`

可选的 neutral 阅读器变量：

- `MUSE_NEUTRAL_API_KEY`
- `MUSE_NEUTRAL_BASE_URL`
- `MUSE_NEUTRAL_MODEL_NAME`

如有任何必需值缺失，API 会返回清晰的配置错误信息，而非虚假的兜底回答。

## 运行时行为

`POST /api/qa`：

1. 系统仅检索截至 `current_chapter` 为止的可见文本。
2. 以问题、高亮和可见上下文查询角色知识库。
3. 构建角色专属的 system prompt。
4. 将角色 prompt 加可见文本证据发送到配置的模型端点。
5. 返回生成的回答以及 `model_name`。

`POST /api/summary`：

1. 系统收集当前章节的可见 chunk。
2. 检索支撑角色证据。
3. 请求配置的角色模型仅对该章节进行摘要。
4. 返回生成的摘要以及 `model_name`。

## 当前接口列表

- `GET /api/personas`
- `GET /api/persona-agents`
- `GET /api/persona-agents/{persona_id}`
- `GET /api/persona-agents/{persona_id}/kb`
- `POST /api/persona-agents/{persona_id}/retrieve`
- `POST /api/persona-agents/{persona_id}/prompt-preview`
- `POST /api/qa`
- `POST /api/summary`

## 注意事项

- 角色模型端点必须是 OpenAI 兼容的。
- 角色风格可以影响解读和语气，但绝不覆盖防剧透边界。
- 角色知识库存储的是结构化证据和摘要，而非完整的受版权保护语料。
