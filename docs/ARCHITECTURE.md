# 架构与逐文件说明

## 1. 为什么这样分层

项目按“入口、接口、核心设施、业务服务、RAG 算法、数据模型”划分。这样做不是为了目录好看，而是为了避免把文件解析、数据库、模型调用和 HTTP 全写进 `main.py`。每一层都能独立测试，也能在不改上层接口的前提下替换实现。

依赖方向如下：

```text
main.py / frontend.py
        ↓
app/main.py → app/api
                 ↓
             app/core/container.py
              ↙       ↘
       app/services    app/rag
              ↘       ↙
              app/models
```

`container.py` 是组装点：负责创建配置、Embedding、ChromaDB、检索器、Ollama 客户端和 Pipeline。业务模块只接收它需要的对象，不自己到处创建全局单例。

## 2. RAG 数据流

### 建库阶段

1. `DocumentManager` 校验并保存上传文件。
2. `DocumentLoader` 按文件类型提取文字和元数据。
3. `TextSplitter` 清洗文字，并生成有重叠的 Chunk。
4. `EmbeddingService` 批量将 Chunk 转为 384 维归一化向量。
5. `ChromaVectorStore` 保存向量、原文和来源元数据。

### 问答阶段

1. `SemanticRetriever` 把问题向量化并查询 Top-K。
2. 候选结果经过最低相似度过滤。
3. `RAGPipeline` 把片段组成有限长度的上下文。
4. `prompt.py` 约束 Qwen 只使用上下文，资料不足时拒答。
5. `OllamaClient` 调用本地模型。
6. Pipeline 返回回答以及结构化 `SourceReference` 列表。

## 3. 每个文件为什么这样命名

### 根目录

| 文件 | 作用与命名原因 |
|---|---|
| `main.py` | ASGI 约定入口；`uvicorn main:app` 表示从 `main` 模块读取 `app` 对象。只做导入，不承载业务。 |
| `frontend.py` | Streamlit 页面入口；名称直接表明这是前端，不与 FastAPI 应用混淆。 |
| `requirements.txt` | Python 社区常见依赖清单名，只列运行项目所需的直接依赖。 |
| `requirements-dev.txt` | `dev` 表示开发环境，在运行依赖之外增加 pytest。 |
| `pytest.ini` | pytest 的标准配置文件名，定义测试目录和 Python 路径。 |
| `.env.example` | 可公开的环境变量模板；`.env` 是本机真实配置且不会提交。 |
| `.gitignore` | 告诉 Git 忽略隐私数据、缓存、日志、模型和虚拟环境。 |
| `.dockerignore` | 告诉 Docker 构建时不要复制无关或敏感文件。 |
| `Dockerfile` | Docker 的默认镜像构建描述文件名。 |
| `README.md` | GitHub 仓库首页；`README` 意为“先读我”。 |

### `app/core`：基础设施

| 文件 | 作用与命名原因 |
|---|---|
| `config.py` | `config` 即配置；用 Pydantic Settings 读取、校验 `.env` 和环境变量。 |
| `logging_config.py` | 名称强调它配置 Python logging，不放业务日志内容；启用控制台和轮转文件日志。 |
| `exceptions.py` | 集中定义领域异常，让底层错误能映射为一致的 API 响应。 |
| `container.py` | “容器”保存应用级依赖并完成对象装配，不是 Docker 容器。 |
| `__init__.py` | 把目录标记为 Python 包；文件可为空。 |

### `app/models`：内部数据契约

| 文件 | 作用与命名原因 |
|---|---|
| `document.py` | 聚合与文档域相关的数据模型：页、Chunk、检索结果、来源、回答和已存文档。 |
| `__init__.py` | 重新导出常用模型，让其他模块可从 `app.models` 简洁导入。 |

核心模型：

- `DocumentPage`：解析后的单页或单个文本单元。
- `TextChunk`：可检索的小片段，继承文档 ID、名称、页码等元数据。
- `SearchResult`：一个 Chunk 加相似度分数。
- `SourceReference`：最终给用户看的来源，不暴露内部路径。
- `RAGAnswer`：回答、是否找到依据和来源列表。
- `StoredDocument`：上传文件在本地管理时的元数据。

### `app/services`：业务服务

| 文件 | 作用与命名原因 |
|---|---|
| `document_loader.py` | `loader` 表示“从外部文件加载为内部模型”；处理 TXT、Markdown 和 PDF。 |
| `document_manager.py` | `manager` 表示文档生命周期管理；负责上传校验、去重、列表、替换和删除。 |
| `knowledge_base.py` | 组合加载、切分、向量化和存储，完成知识库构建。 |
| `ollama_client.py` | `client` 表示外部服务客户端；封装健康检查、HTTP 调用、超时和错误。 |
| `__init__.py` | 标记服务包。 |

### `app/rag`：检索增强生成核心

| 文件 | 作用与命名原因 |
|---|---|
| `text_splitter.py` | 手写文本清洗和滑动窗口切分；`splitter` 的职责就是把长文本变成 Chunk。 |
| `embeddings.py` | 将文本批量编码为向量；复数名表示向量表示这一类能力。 |
| `vector_store.py` | 封装 ChromaDB 的写入、查询、计数、删除和重建。 |
| `retriever.py` | `retriever` 意为检索器；负责问题向量化、Top-K 和阈值过滤。 |
| `prompt.py` | 集中保存和构建提示词，防止提示词散落在业务代码中。 |
| `pipeline.py` | `pipeline` 表示有顺序的数据处理管线；把检索、上下文、生成和引用串起来。 |
| `__init__.py` | 标记 RAG 包。 |

### `app/api` 与应用创建

| 文件 | 作用与命名原因 |
|---|---|
| `api/schemas.py` | `schema` 是 HTTP 数据结构契约；定义请求体和响应体，而不是数据库表。 |
| `api/routes.py` | `route` 是 URL 与处理函数的映射；集中 REST 接口。 |
| `api/__init__.py` | 标记 API 包。 |
| `app/main.py` | 创建 FastAPI 实例、生命周期、异常处理器和路由；与根入口分开便于测试。 |
| `app/__init__.py` | 标记整个应用包。 |

### 脚本、测试和数据

| 文件/目录 | 作用与命名原因 |
|---|---|
| `scripts/search.py` | 不接大模型的命令行检索实验，证明“检索”和“生成”是两部分。 |
| `scripts/evaluate.py` | 从带标签数据集计算可复现指标，可选调用 Ollama 检查生成结果。 |
| `tests/test_config.py` | 配置默认值、环境变量和边界校验测试。 |
| `tests/test_document_loader.py` | 文本/PDF 解析、编码和异常测试。 |
| `tests/test_text_splitter.py` | 清洗、Chunk 大小、重叠和元数据继承测试。 |
| `tests/test_embedding_and_retrieval.py` | 向量维度、归一化、存储和语义检索测试。 |
| `tests/test_rag_pipeline.py` | 有答案、无答案和来源引用的 Pipeline 单元测试。 |
| `tests/test_api.py` | 健康、上传、构建、更新、删除、校验和问答 API 集成测试。 |
| `data/examples/*.md` | 可公开的员工、财务和安全示例资料，便于复现实验。 |
| `evaluation/dataset.json` | 问题、预期来源、是否应回答和关键词组成的标签集。 |
| `evaluation/retrieval_baseline.json` | 记录一次真实运行环境、参数、指标和限制，避免伪造结果。 |

## 4. 关键设计选择

- 使用 `pathlib.Path`：跨平台表达路径，少写字符串拼接和反斜杠转义。
- 使用 SHA-256 文档 ID：内容相同即 ID 相同，可用于去重和追踪向量来源。
- 手写字符切分：更容易看到 Chunk 和 overlap 的真实作用；生产中可换成标题感知或 Token 感知切分。
- E5 前缀：文档用 `passage:`、问题用 `query:`，符合该模型的训练方式。
- 向量归一化：归一化后点积等价于余弦相似度，分数更容易比较。
- 设置拒答阈值：Top-K 只表示“最像的几个”，不保证它们真的相关。
- ChromaDB 持久化：重启 API 后向量仍存在；评估脚本则用内存集合避免污染业务数据。
- 不上传本地数据：Git 仓库只放演示数据，真实企业文件、日志和向量库都被忽略。

## 5. 可替换点

当前接口边界允许逐步替换：Embedding 模型、向量数据库、Ollama 模型、切分策略和 UI 都可以单独变化。替换后必须重新跑测试与评估，因为模型维度、相似度分布、延迟和阈值会改变。
