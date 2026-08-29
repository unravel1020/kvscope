# RESEARCH — 研究思路全程记录（kvscope 诞生记）

> 记录从「想要一个有技术壁垒的开源项目」到「kvscope 六版本完成」的完整决策链。
> 日期：2026-08-28 ｜ 状态：v0.6 完成，持续演进

## 一、起点：为什么不做"工具层"项目

**筛选标准（与 mentor 讨论后定）**：
1. AI 本身不能直接替代（需要程序提供新能力/数据/执行环境/性能）
2. 有明确技术壁垒（性能、kernel、调度、KV Cache、量化、分布式、runtime、benchmark）
3. 个人开发者能切入（不正面撞 PyTorch/vLLM/SGLang/llama.cpp，找"缝"）
4. 能形成技术履历（"我解决了一个 AI Infra 问题"而非"我会用 AI 写工具"）

**被否决的方向**：agent-doctor（环境检查脚本，AI 自己能做）、
agent-replay（日志解析 + Web UI，沦为"漂亮日志查看器"）、
普通 skill 工具——**核心风险：AI 本身就是它的替代品**。

## 二、调研阶段（3 条并行子代理 + 主代理一手验证）

### 调研线 1：TraceLab 论文（arXiv 2606.30560，UW SyFI）

真实编码 agent 负载 trace：43 人 8 个月 Claude Code/Codex，357K LLM 步骤。

**关键发现**：
- 每步中位 prefix 119K / append 875 / output 214 tokens（前缀远大于追加）
- 全局 prefix cache 命中 95.7%，真正 fresh 仅 19%，prefill 放大 5.3×
- 会话 92.3% 时间在等人类，间隙 >5 分钟缓存开始 miss
- 工具 top3 占 80%+ 调用，但 >1 分钟调用占 4.9% 却耗 92% 工具时间

**留给后来者的缝**：trace→引擎 replay、跨后端对比、KV 字节级仿真、
本地开源模型场景、gap-duration 预测器（论文点名）。

### 调研线 2：SGLang profiling/可观测性现状

官方已有（比预期完整）：PyTorch Profiler、Nsight/NVTX、OTel trace、
bench_serving（含 agentic-trace 数据集）、RequestMetricsExporter。

**关键缺口**（有 issue/PR 证据）：
- Engine 已无 `get_kv_cache_usage`；无 per-request KV 分配量
- OpenAI 兼容 API 零 per-request 指标（#36678 open）
- kv_cache_usage_perc 对标 vLLM 缺失（#5979 挂三年）
- **radix 树内部结构零导出**（节点/深度/eviction 明细）

### 调研线 3：KV cache 工具盘点

- 理论计算器已卷烂（无护城河）
- 运行时分析器稀缺：kvcachescope（2★ 未验证）、vLLM-PrefixCache-Monitor（13★）
- **"radix 树结构/共享/碎片"一行 = 全行业唯一整行空白**
- vLLM 官方离线 analyzer 在 PR（只做 vLLM、只做跑前预估）

### 方向取舍结论

| 方向 | 判决 | 理由 |
|------|------|------|
| KV Cache 分析器 | ✅✅ 最优 | 全行业空白 + 学习即项目 + 白天无 GPU 可开发 |
| Inference Profiler | ✅ 缩窄 | 官方做了单请求分解，跨请求聚合是空白 |
| Agent Benchmark | ✅ 有竞争 | TraceLab 开源了数据但 NVIDIA aiperf 在填 |
| Tool-use eval | ⏸ | 未深入调研 |
| Scheduling | ❌ 出局 | SGLang 已有 schedule_simulator（#33824） |

## 三、项目演进（6 个版本，一天完成）

```
v0.1 快照分析    结构/共享/碎片/驱逐      analyze 子命令
  ↓ 用户："继续"
v0.2 端到端       SGLang 真实 radix（create_simulated）→ dump → analyze
  ↓
v0.3 per-turn     KV-reuse 命中率曲线      turns 子命令
  ↓
v0.4 事件流       churn/驱逐时间线          events 子命令
  ↓
v0.5 gap 预测     TraceLab 方向：间隙→命中率衰减 + logistic 模型   gaps 子命令
  ↓
v0.6 驱逐对比     CacheWise 方向：LRU vs 预测性，-29% 驱逐       evict 子命令
```

### 每个版本的核心突破

- **v0.2**：发现 `create_simulated()` = 纯 CPU 跑真实 SGLang radix 的官方入口，
  从此白天就能端到端验证
- **v0.3**：`full_kv_hit_length` 在模拟下恒 0 → 发明"parent 链累加"算法
- **v0.4**：native hash 扩展坑 → 自实现确定性哈希（格式对齐官方）
- **v0.5**：`last_access_time` 是真实时钟不可模拟比较 → "主动清理"策略；
  实证复现 TraceLab"间隙>5分钟缓存失效"
- **v0.6**：SGLang `evict()` 按节点驱逐 → 有限池实验设计；
  **实证：预测性驱逐同命中率下 -29% 驱逐操作**

## 四、与学术界的关系

| 论文/工作 | 我们的对应 |
|---|---|
| TraceLab（负载刻画） | v0.3 turns + v0.5 gaps 复现其发现 |
| CacheWise（预测性驱逐） | v0.6 evict 实验（可复现证据） |
| vLLM 官方 analyzer（#48369） | 模仿其 CLI/报告模式，但覆盖 SGLang + 跑后结构 |

## 五、后续方向（v0.7+ 候选）

1. **真实 server dump hook**：接 `KVCacheEventRecorder.take()` 或运行时遍历，
   晚上 RTX 3080 验证真实服务器
2. **跨框架对比**：加 vLLM 数据源（用官方 helper 而非重实现 hash）
3. **gap 模型 → CacheWise 全链路**：预测性驱逐决策直接消费 gaps 模型的
   cache_decay_probability
4. **上游 PR**：补 SGLang 官方缺口（#27701 kv_cache_usage_perc / #36678
   OpenAI per-request metrics），以 PR 建立社区声望，反哺本项目

## 六、经验教训（踩坑记录）

1. **API 版本漂移**：SGLang 迭代快，`InsertParams.key` 要 RadixKey、
   `InsertResult` 字段名变了——拉代码前先看 base_prefix_cache.py 签名
2. **native 扩展依赖**：`enable_kv_cache_events` 触发 C++ 扩展，纯 CPU 用
   自实现替代（格式对齐即可）
3. **真实时钟 vs 模拟时钟**：SGLang 内部 `time.monotonic()` 不可与模拟时钟比较
4. **驱逐粒度**：SGLang `evict()` 按节点驱逐（整叶子），设计实验时控制叶子大小
5. **锁保护**：`create_simulated` 不锁共享前缀，需手动 `inc_lock_ref`
6. **PowerShell 引号**：多行 bash -c 会被 PowerShell 解析，用脚本文件方式
