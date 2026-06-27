# ReMe 项目复现 Plan

> 基于 8卡3090 资源的完整全流程复现方案

---

## 一、资源评估与可行性分析

### 1.1 硬件资源

| 资源 | 规格 | 可用用途 |
|------|------|----------|
| **GPU** | 8x RTX 3090 (24GB VRAM each) | 本地LLM、本地Embedding |
| **总显存** | 192GB | 可部署7B-13B模型 |
| **内存** | 假设128GB+ | 索引缓存、文件监控 |
| **存储** | 假设2TB+ SSD | workspace、索引持久化 |

### 1.2 ReMe 组件算力需求分析

| 组件 | 默认配置 | GPU需求 | 本地部署方案 |
|------|----------|---------|--------------|
| **LLM** | API调用 (qwen3.7-plus) | 无 | 本地部署Qwen-7B/14B |
| **Embedding** | API调用 (text-embedding-v4) | 无 | 本地部署BGE-base/large |
| **BM25索引** | CPU计算 | 无 | 无需GPU |
| **文件监控** | CPU + 磁盘I/O | 无 | 无需GPU |
| **图谱存储** | 内存 + 磁盘 | 无 | 无需GPU |

### 1.3 复现策略选择

**策略A：API模式（推荐先用这个跑通全流程）**
- 优点：无需本地GPU，快速验证功能
- 缺点：有API成本，有网络延迟
- 适用：功能验证、流程理解

**策略B：本地部署模式（深入学习）**
- 优点：无API成本，可控性强
- 缺点：需要GPU资源，需要模型部署经验
- 适用：深入理解、性能优化

**策略C：混合模式（生产环境推荐）**
- LLM：本地部署小模型 + API调用大模型
- Embedding：本地部署
- 适用：成本控制、性能平衡

---

## 二、Phase 1：API模式全流程复现（1-2天）

### 2.1 环境准备

```bash
# 1. 克隆项目
cd /home/chenyizhou
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe

# 2. 创建虚拟环境
conda create -n reme python=3.11 -y
conda activate reme

# 3. 安装依赖
pip install -e ".[core]"

# 4. 配置环境变量
cat > .env <<'EOF'
EMBEDDING_API_KEY=your_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL_NAME=text-embedding-v4
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus
EOF
```

### 2.2 启动服务

```bash
# 启动ReMe服务
reme start

# 检查服务状态
reme version
reme health_check
```

### 2.3 功能验证清单

#### 2.3.1 基础功能

```bash
# 1. 写入记忆
reme write path="daily/2026-06-27/test.md" name="测试记忆" description="这是一个测试" content="# 测试\n\n这是测试内容。"

# 2. 读取记忆
reme read path="daily/2026-06-27/test.md"

# 3. 搜索记忆
reme search query="测试" limit=5

# 4. 编辑记忆
reme edit path="daily/2026-06-27/test.md" old="测试内容" new="修改后的内容"

# 5. 删除记忆
reme delete path="daily/2026-06-27/test.md"
```

#### 2.3.2 Auto Memory 功能

```bash
# 模拟对话记忆
reme auto_memory messages='[{"role":"user","content":"我喜欢Python 3.11"},{"role":"assistant","content":"好的，我会记住您喜欢Python 3.11"}]' session_id="test-session-001"
```

#### 2.3.3 Auto Dream 功能

```bash
# 运行auto_dream
reme auto_dream date="2026-06-27"

# 查看生成的digest
ls -la digest/personal/
ls -la digest/procedure/
ls -la digest/wiki/

# 查看interests
cat daily/2026-06-27/interests.yaml
```

#### 2.3.4 Proactive 功能

```bash
# 读取主动提醒
reme proactive date="2026-06-27"
```

### 2.4 验证指标

| 功能 | 验证方法 | 预期结果 |
|------|----------|----------|
| **写入** | 写入后读取 | 内容一致 |
| **搜索** | 搜索已写入内容 | 返回相关结果 |
| **Auto Memory** | 调用后检查daily | 生成daily note |
| **Auto Dream** | 运行后检查digest | 生成digest节点 |
| **Proactive** | 读取interests | 返回主题列表 |

---

## 三、Phase 2：本地部署模式（3-5天）

### 3.1 本地Embedding部署

#### 3.1.1 模型选择

| 模型 | 参数量 | 显存占用 | 维度 | 推荐 |
|------|--------|----------|------|------|
| **bge-base-zh-v1.5** | 102M | ~2GB | 768 | ⭐ 推荐 |
| **bge-large-zh-v1.5** | 326M | ~4GB | 1024 | 高精度 |
| **bge-m3** | 568M | ~6GB | 1024 | 多语言 |

#### 3.1.2 部署步骤

```bash
# 1. 安装依赖
pip install sentence-transformers fastapi uvicorn

# 2. 创建Embedding服务
cat > embedding_server.py <<'EOF'
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
import torch

app = FastAPI()

# 加载模型
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
model = model.to("cuda:0")

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str = "bge-base-zh-v1.5"

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    embeddings = model.encode(request.input, normalize_embeddings=True)
    return {
        "data": [
            {
                "embedding": emb.tolist(),
                "index": i
            }
            for i, emb in enumerate(embeddings)
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
EOF

# 3. 启动服务
python embedding_server.py --port 8080
```

#### 3.1.3 配置ReMe使用本地Embedding

```bash
# 更新.env
cat > .env <<'EOF'
EMBEDDING_API_KEY=not-needed
EMBEDDING_BASE_URL=http://localhost:8080/v1
EMBEDDING_MODEL_NAME=bge-base-zh-v1.5
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus
EOF
```

### 3.2 本地LLM部署

#### 3.2.1 模型选择

| 模型 | 参数量 | 显存占用 | 推荐 |
|------|--------|----------|------|
| **Qwen2-7B-Instruct** | 7B | ~14GB | ⭐ 单卡可用 |
| **Qwen2-14B-Instruct** | 14B | ~28GB | 双卡并行 |
| **Qwen2-72B-Instruct** | 72B | ~144GB | 8卡并行 |

#### 3.2.2 使用vLLM部署

```bash
# 1. 安装vLLM
pip install vllm

# 2. 启动服务（单卡7B）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 3. 启动服务（双卡14B）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-14B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 4. 启动服务（8卡72B）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-72B-Instruct \
    --tensor-parallel-size 8 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

#### 3.2.3 配置ReMe使用本地LLM

```bash
# 更新.env
cat > .env <<'EOF'
EMBEDDING_API_KEY=not-needed
EMBEDDING_BASE_URL=http://localhost:8080/v1
EMBEDDING_MODEL_NAME=bge-base-zh-v1.5
LLM_API_KEY=not-needed
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL_NAME=Qwen/Qwen2-7B-Instruct
EOF
```

### 3.3 GPU分配方案

#### 3.3.1 方案A：单卡测试（快速验证）

```text
GPU 0: Embedding (bge-base, ~2GB)
GPU 1-7: 空闲（可用于其他任务）
LLM: 使用API
```

#### 3.3.2 方案B：双卡部署（平衡方案）

```text
GPU 0: Embedding (bge-base, ~2GB)
GPU 1-2: LLM (Qwen2-7B, ~14GB)
GPU 3-7: 空闲
```

#### 3.3.3 方案C：全卡部署（性能最大化）

```text
GPU 0: Embedding (bge-base, ~2GB)
GPU 1-8: LLM (Qwen2-72B, ~144GB, tensor-parallel)
```

#### 3.3.4 方案D：多实例部署（并行测试）

```text
GPU 0-1: Embedding + LLM实例1 (Qwen2-7B)
GPU 2-3: LLM实例2 (Qwen2-7B)
GPU 4-5: LLM实例3 (Qwen2-7B)
GPU 6-7: LLM实例4 (Qwen2-7B)
```

---

## 四、Phase 3：深入学习与优化（5-7天）

### 4.1 代码结构学习路径

```text
1. 入口层
   reme/reme.py → CLI入口
   reme/application.py → 应用装配

2. 组件层
   reme/components/file_store/ → 文件存储
   reme/components/keyword_index/ → BM25索引
   reme/components/embedding_store/ → Embedding存储
   reme/components/file_graph/ → Wikilink图谱

3. Step层
   reme/steps/index/search.py → 搜索实现
   reme/steps/evolve/auto_memory.py → 自动记忆
   reme/steps/evolve/dream/ → Auto Dream

4. 配置层
   reme/config/default.yaml → 默认配置
   reme/config/config_parser.py → 配置解析
```

### 4.2 关键算法深入学习

#### 4.2.1 BM25 索引实现

```python
# 学习要点：
# 1. 倒排索引结构
# 2. IDF计算公式
# 3. lazy deletion机制
# 4. 持久化策略

# 实践：
# 1. 阅读 reme/components/keyword_index/bm25_index.py
# 2. 运行测试 tests/unit/test_keyword_index.py
# 3. 修改参数观察效果
```

#### 4.2.2 RRF 融合算法

```python
# 学习要点：
# 1. RRF公式：score = w1/(K+rank1) + w2/(K+rank2)
# 2. K=60的选择原因
# 3. 向量权重和关键词权重的平衡

# 实践：
# 1. 阅读 reme/steps/index/search.py
# 2. 修改vector_weight观察结果变化
# 3. 对比纯BM25和纯向量的效果
```

#### 4.2.3 Auto Dream 流程

```python
# 学习要点：
# 1. Extract: LLM抽取memory units
# 2. Integrate: Agent整合到digest
# 3. Topics: 生成主动提醒
# 4. Finish: checkpoint管理

# 实践：
# 1. 阅读 reme/steps/evolve/dream/ 目录
# 2. 运行测试 tests/unit/test_auto_dream.py
# 3. 手动调用auto_dream观察输出
```

### 4.3 实验设计

#### 4.3.1 检索质量实验

```python
# 实验1：BM25 vs 向量 vs 混合检索
# 变量：vector_weight = [0.0, 0.3, 0.5, 0.7, 1.0]
# 指标：Precision@5, Recall@5, MRR

# 实验2：chunk大小对检索的影响
# 变量：chunk_size = [100, 200, 500, 1000]
# 指标：检索精度、响应时间

# 实验3：链接展开的效果
# 变量：expand_links = [True, False]
# 指标：结果多样性、相关性
```

#### 4.3.2 Auto Dream 质量实验

```python
# 实验1：Extract的unit质量
# 方法：人工评估抽取的units是否准确
# 指标：准确率、召回率

# 实验2：Integrate的整合质量
# 方法：对比daily和digest的内容
# 指标：信息保留率、去重率

# 实验3：Topics的相关性
# 方法：评估生成的interests是否相关
# 指标：相关性评分
```

### 4.4 性能优化实践

#### 4.4.1 索引优化

```python
# 1. BM25索引优化
# - 批量添加文档
# - 定期optimize_index
# - 调整k1和b参数

# 2. Embedding缓存
# - 启用LRU缓存
# - 调整缓存大小
# - 持久化缓存

# 3. 文件监控优化
# - 调整quiet_window
# - 批量处理变化
# - 异步更新索引
```

#### 4.4.2 并发优化

```python
# 1. 异步调用
# - LLM调用异步化
# - Embedding批量异步
# - 文件I/O异步

# 2. 并发控制
# - 文件级锁
# - 索引更新队列
# - 限流控制
```

---

## 五、Phase 4：扩展与创新（7-10天）

### 5.1 功能扩展

#### 5.1.1 多模态记忆

```python
# 扩展支持图片记忆
# 1. 图片Embedding（CLIP模型）
# 2. 图片描述生成（多模态LLM）
# 3. 图片检索
```

#### 5.1.2 知识图谱增强

```python
# 增强wikilink图谱
# 1. 自动关系抽取
# 2. 图谱可视化
# 3. 图谱推理
```

#### 5.1.3 记忆衰减机制

```python
# 实现记忆衰减
# 1. 时间衰减函数
# 2. 访问频率加权
# 3. 重要性评分
```

### 5.2 性能优化

#### 5.2.1 大规模数据优化

```python
# 1. 分布式索引
# - 分片策略
# - 一致性哈希
# - 负载均衡

# 2. 缓存优化
# - 多级缓存
# - 缓存预热
# - 缓存淘汰策略
```

#### 5.2.2 检索优化

```python
# 1. ANN索引
# - FAISS集成
# - HNSW索引
# - 量化压缩

# 2. 重排序
# - Cross-encoder重排
# - 学习型排序
# - 个性化排序
```

### 5.3 业务场景适配

#### 5.3.1 个人助理场景

```python
# 1. 用户画像构建
# 2. 偏好学习
# 3. 主动推荐
```

#### 5.3.2 编程助手场景

```python
# 1. 代码风格记忆
# 2. 项目上下文理解
# 3. 错误模式学习
```

#### 5.3.3 知识管理场景

```python
# 1. 文档自动整理
# 2. 知识关联发现
# 3. 知识更新提醒
```

---

## 六、学习资源与参考

### 6.1 核心文档

| 文档 | 路径 | 内容 |
|------|------|------|
| **快速开始** | docs/zh/quick_start.md | 安装、配置、启动 |
| **代码框架** | docs/zh/framework.md | 架构设计、分层职责 |
| **Memory as File** | docs/zh/memory_as_file.md | 文件格式、语义 |
| **Memory Search** | docs/zh/memory_search.md | 检索机制、索引构建 |
| **Auto Memory** | docs/zh/auto_memory.md | 对话记忆流程 |
| **Auto Dream** | docs/zh/auto_dream.md | 长期记忆沉淀 |

### 6.2 关键代码文件

| 文件 | 内容 | 学习重点 |
|------|------|----------|
| `reme/components/keyword_index/bm25_index.py` | BM25实现 | 倒排索引、IDF计算 |
| `reme/steps/index/search.py` | 搜索实现 | RRF融合、链接展开 |
| `reme/steps/evolve/dream/` | Auto Dream | 抽取、整合、主题 |
| `reme/components/embedding_store/` | Embedding存储 | 缓存、持久化 |
| `reme/config/default.yaml` | 默认配置 | 组件配置、Job编排 |

### 6.3 测试文件

| 测试 | 路径 | 内容 |
|------|------|------|
| **BM25测试** | tests/unit/test_keyword_index.py | 索引功能 |
| **搜索测试** | tests/unit/test_search_step.py | 搜索流程 |
| **Auto Dream测试** | tests/unit/test_auto_dream.py | Dream流程 |
| **文件图谱测试** | tests/unit/test_file_graph.py | 图谱操作 |

---

## 七、时间规划

### 7.1 总体时间安排

| 阶段 | 时间 | 目标 | 产出 |
|------|------|------|------|
| **Phase 1** | 1-2天 | API模式全流程复现 | 功能验证报告 |
| **Phase 2** | 3-5天 | 本地部署模式 | 本地服务部署 |
| **Phase 3** | 5-7天 | 深入学习与优化 | 实验报告、优化方案 |
| **Phase 4** | 7-10天 | 扩展与创新 | 扩展功能、性能提升 |

### 7.2 每日任务

#### Day 1-2: Phase 1

```text
Day 1:
- 上午：环境准备、项目克隆、依赖安装
- 下午：启动服务、基础功能验证

Day 2:
- 上午：Auto Memory、Auto Dream验证
- 下午：Proactive验证、问题排查
```

#### Day 3-5: Phase 2

```text
Day 3:
- 上午：Embedding模型选择、服务部署
- 下午：配置ReMe使用本地Embedding

Day 4:
- 上午：LLM模型选择、vLLM安装
- 下午：LLM服务部署、配置

Day 5:
- 上午：全链路测试、性能测试
- 下午：问题排查、优化调整
```

#### Day 6-10: Phase 3-4

```text
Day 6-7:
- 上午：代码结构学习
- 下午：关键算法深入

Day 8-9:
- 上午：实验设计、执行
- 下午：结果分析、优化

Day 10:
- 上午：扩展功能开发
- 下午：总结、文档整理
```

---

## 八、风险与应对

### 8.1 技术风险

| 风险 | 影响 | 应对方案 |
|------|------|----------|
| **API不可用** | 无法使用API模式 | 优先本地部署 |
| **GPU显存不足** | 无法部署大模型 | 使用小模型或量化 |
| **模型效果差** | 检索/生成质量低 | 调整参数、更换模型 |
| **性能瓶颈** | 响应慢 | 优化索引、缓存 |

### 8.2 时间风险

| 风险 | 影响 | 应对方案 |
|------|------|----------|
| **环境配置耗时** | 延缓进度 | 提前准备、参考文档 |
| **问题排查耗时** | 延缓进度 | 记录问题、寻求帮助 |
| **实验设计不合理** | 结果不可靠 | 参考论文、设计验证 |

---

## 九、成功标准

### 9.1 Phase 1 成功标准

- [ ] 服务正常启动
- [ ] 基础CRUD功能正常
- [ ] Auto Memory生成daily note
- [ ] Auto Dream生成digest节点
- [ ] Proactive返回interests

### 9.2 Phase 2 成功标准

- [ ] 本地Embedding服务正常
- [ ] 本地LLM服务正常
- [ ] ReMe使用本地服务正常工作
- [ ] 性能满足需求

### 9.3 Phase 3 成功标准

- [ ] 理解代码架构
- [ ] 理解关键算法
- [ ] 完成实验验证
- [ ] 提出优化方案

### 9.4 Phase 4 成功标准

- [ ] 实现扩展功能
- [ ] 性能提升可量化
- [ ] 业务场景适配
- [ ] 文档完整

---

## 附录：常用命令

### A.1 ReMe 命令

```bash
# 启动服务
reme start

# 检查状态
reme version
reme health_check

# 记忆操作
reme write path="..." name="..." description="..." content="..."
reme read path="..."
reme edit path="..." old="..." new="..."
reme delete path="..."

# 搜索
reme search query="..." limit=5

# 自动记忆
reme auto_memory messages='[...]' session_id="..."
reme auto_dream date="YYYY-MM-DD"
reme proactive date="YYYY-MM-DD"
```

### A.2 服务管理

```bash
# 查看日志
tail -f ~/.reme/logs/reme.log

# 停止服务
pkill -f "reme start"

# 重启服务
pkill -f "reme start" && reme start
```

### A.3 测试命令

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/

# 运行特定测试
pytest tests/unit/test_keyword_index.py
```

---

*本Plan基于8卡3090资源制定，可根据实际情况调整*
