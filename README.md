# Milvus FAQ Retrieval System

一个基于 Milvus 和 LlamaIndex 构建的智能 FAQ 检索系统。

## 功能特点

- **自然语言检索**：输入用户问题（如“如何退货？”），系统通过语义搜索返回最相关的 FAQ 条目。
- **热更新知识库**：提供 API 接口实时摄入新的 FAQ 数据，自动建立索引。
- **高性能向量库**：使用 Milvus 作为后端向量数据库，支持海量数据的高效检索。
- **文档切片优化**：内置语义切分与重叠策略（Chunking & Overlap），提升长文本的检索准确率。
- **Web 测试界面**：提供 Vue.js 构建的简洁聊天界面，方便直观测试。

## 技术栈

- **Web 框架**: FastAPI
- **向量数据库**: Milvus (Standalone)
- **RAG 框架**: LlamaIndex
- **Embedding 模型**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **前端**: Vue.js 3 + Tailwind CSS

## 快速开始

### 1. 环境准备

确保已安装 Docker 和 Python 3.10+。

启动 Milvus 服务：
```bash
# 假设你已有 Milvus 的 docker-compose.yml
docker compose up -d
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

为了加速模型下载，建议设置 HF 镜像环境变量：

```bash
HF_ENDPOINT=https://hf-mirror.com uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务启动后，访问 `http://localhost:8000` 即可看到测试界面。

API 文档地址：`http://localhost:8000/docs`

## API 使用说明

### 1. 摄入知识 (Ingest)

**POST** `/ingest`

```json
{
  "faqs": [
    {
      "question": "如何退货？",
      "answer": "请在订单页面点击退货按钮，并填写退货原因。"
    },
    {
      "question": "退款多久到账？",
      "answer": "退款通常在商家确认收货后1-3个工作日内原路返回。"
    }
  ]
}
```

### 2. 搜索 (Search)

**POST** `/search`

```json
{
  "query": "怎么退货",
  "top_k": 3
}
```

## 项目结构

```
.
├── app/
│   ├── core/
│   │   ├── config.py         # 配置项
│   │   ├── milvus_client.py  # Milvus 连接与设置
│   │   ├── ingestion.py      # 文档摄入逻辑 (含分片优化)
│   │   └── retrieval.py      # 检索逻辑
│   ├── models/
│   │   └── schemas.py        # Pydantic 数据模型
│   └── main.py               # FastAPI 主程序
├── static/                   # 前端静态资源
├── requirements.txt          # 项目依赖
└── verify.py                 # 验证脚本
```

## 许可证

MIT License
