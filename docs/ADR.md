# ADR — 架构决策记录（kvscope）

> 记录关键架构决策及其背景、权衡、后果。按时间顺序。
> 日期：2026-08-28（一天内完成 v0.1 → v0.6）

## ADR-001：项目定位 = SGLang 的 KV cache 显微镜（离线分析）

**状态**：已采纳（v0.1）

**背景**：个人开发者在 SGLang（主攻）/llama.cpp（辅修）学习路线中，希望做一个
"AI 无法替代的"开源项目——需要程序提供新能力/数据/执行环境，而非"AI 能帮你做的工具"。

**决策**：做 **SGLang KV Cache 分析器**——离线分析 radix cache 的
结构/共享率/碎片/驱逐效率，定位"跑起来之后"的真实结构分析。

**依据**（调研结论，见 docs/RESEARCH.md）：
- SGLang 官方只有 `cache_hit_rate` 一个聚合指标，radix 树内部结构零导出
- vLLM 官方在做"跑之前"的离线 workload 预演（#47993/#48369/#48838），
  **只覆盖 vLLM、只做跑前预估**——"跑后结构分析 + SGLang 侧"是无人区
- kvcachescope（最近似工具）仅 2★、出生 <1 月、hook 能力未验证
- 与 mem_cache 学习主线完全重合：学习即开发

**后果**：项目成为"读快照→离线分析"的工具，不依赖 GPU 可白天开发。

## ADR-002：数据模型 = 快照 dump（JSONL），非实时 hook

**状态**：已采纳（v0.1）

**背景**：分析器需要数据源。两个候选：运行时 hook（侵入式）vs 快照 dump（解耦式）。

**决策**：**离线快照 dump → CPU 分析**解耦模式。快照为 JSONL：
`{"type":"snapshot", "version":1, "page_size":1, "eviction_policy":"lru",
 "pool":{...}, "nodes":[{"id","parent","token_len","children","lock_ref","hit_count","priority","evicted","last_access"}]}`

**权衡**：
- hook 模式数据更全，但侵入引擎、稳定性争议、API 漂移
- dump 模式非侵入、可离线、白天无 GPU 可开发；代价是只能拿到 dump 时刻的结构

**后果**：快照格式版本化（version 字段），分析端不依赖 SGLang 安装（duck-typed）。

## ADR-003：模仿 vLLM analyzer 的"输入输出模式"，但算法用 SGLang 语义

**状态**：已采纳（v0.1）

**背景**：用户要求"模仿 vLLM 成熟做法"。vLLM analyzer（PR #48369）是
JSONL 输入 + CLI + text/json 双格式 + 防双重计数 + 合成数据单测。

**决策**：
- **模仿**：CLI 子命令模式、AnalysisReport dataclass（to_dict/render_text）、
  合成快照单测策略、防双重计数思想
- **不模仿**：vLLM 的 block-hash 链（SGLang 是 token 级 radix 树，无块哈希概念），
  改为直接解析 radix 树快照

**后果**：代码结构熟悉 vLLM 的读者易读，但核心算法是 SGLang 原生的。

## ADR-004：复用 SGLang 真实 radix 代码做端到端（create_simulated）

**状态**：已采纳（v0.2）

**背景**：如何验证分析器？起真实服务器需要 GPU/模型，白天不可行。

**决策**：用 SGLang 官方 `RadixCache.create_simulated()`（"Init a radix cache
without memory pools for simulation purpose"）——**纯 CPU 跑真实 radix 代码**，
配合 `mock_allocator=Mock()` 支持 evict。

**发现**（v0.2 过程）：
- `InsertParams.key` 必须是 `RadixKey`（非裸 array）
- `InsertResult` 字段：`prefix_len`/`total_len`/`last_device_node`
- `create_simulated` 的 root 不锁子节点，系统提示词需手动 `inc_lock_ref` 保护

**后果**：白天 WSL 即可在真实 SGLang 代码上复现任意工作负载，端到端验证可行。

## ADR-005：per-turn 命中长度 = 沿 last_device_node parent 链累加

**状态**：已采纳（v0.3）

**背景**：多轮 agent 工作负载需要每轮命中数据。`MatchResult.full_kv_hit_length`
在纯模拟下恒为 0（它服务 HiCache 场景）。

**决策**：用 `compute_hit_length_from_node(node)`——沿 `last_device_node` 的
parent 链累加 `token_len`，即从 root 到匹配节点的路径总长 = 精确命中前缀长度。
实测：全命中 100→100、分裂后部分命中 110→110、冷 miss 0。

**后果**：不依赖 SGLang 内部字段名，任何暴露 TreeNode 形状的对象都能算。

## ADR-006：事件流 = 自实现确定性哈希（避开 native 扩展）

**状态**：已采纳（v0.4）

**背景**：`enable_kv_cache_events=True` 会触发 HiCache native hash C++ 扩展
（`compute_node_hash_values` → native_hash），纯 CPU 环境未编译无法运行。

**决策**：事件**结构**对齐 SGLang `BlockStored`/`BlockRemoved`/`AllBlocksCleared`
wire 格式（block_hashes/parent_block_hash/medium），但哈希用纯 Python 确定性
实现（sha256 截断）替代 native。

**后果**：kvscope 消费的格式与官方兼容（未来接 dynamo 等 KV-aware router 可复用），
且纯 CPU 可跑。

## ADR-007：gap 衰减模型 = 模拟时钟 + 主动清理策略

**状态**：已采纳（v0.5）

**背景**：复现 TraceLab"间隙>5分钟缓存失效"。但 SGLang 的
`node.last_access_time` 用 `time.monotonic()`（真实时钟），**不可与模拟时钟比较**。

**决策**：
- 间隙用模拟时钟（`clock += gap`）
- 长间隙后**主动清空 evictable_leaves**（模拟"该会话缓存已冷"），
  而非按 last_access_time 过滤
- 拟合 logistic 模型 `P(hit|gap) = 1/(1+exp(-(a+b·ln(gap))))`

**后果**：v0.5 实证复现 TraceLab：短间隙命中 93%、长间隙后 0%。

## ADR-008：驱逐策略对比 = 有限池 + 主动清理为唯一差异

**状态**：已采纳（v0.6）

**背景**：验证预测性驱逐（CacheWise 方向）优于 LRU。踩坑：
SGLang `evict()` **按节点驱逐非 token 数**（一次清空整个叶子子树）、
指定 victim 不可行（evict 内部按 LRU 堆弹）。

**决策**：两种策略在**完全相同**的请求序列 + 有限 KV 池（1000 tokens）下：
- LRU 基线：只在池满时 `cache.evict()`（SGLang 原生 LRU）
- 预测性：长间隙后主动清理 evictable_leaves + 池满时同样 LRU
- **唯一差异 = 主动清理**

**结果**：相同命中率（88%）下预测性驱逐**减少 29% 驱逐轮次**（59→42）。

**后果**：可复现的 CacheWise 式实验证据。

## ADR-009：零运行时依赖（duck-typing）

**状态**：已采纳（v0.1 起）

**背景**：分析端不应强制用户安装 SGLang（用户可能只有快照文件）。

**决策**：`kvscope/` 包零依赖（纯 stdlib），SGLang 只在 `TYPE_CHECKING` 下导入；
`dump.py` 用 duck-typing 读取 TreeNode 形状。唯一依赖在 `scripts/`（模拟脚本，需 sglang）。

**后果**：`pip install kvscope` 即用，测试无 tokenizer/SGLang 依赖。
