# KVScope — SGLang KV Cache 显微镜

> 离线分析 SGLang radix cache 的结构、共享率、碎片与驱逐效率。
> 定位：SGLang 的 "KV cache 显微镜"——读运行时快照，回答缓存内部发生了什么。
>
> 研究背景与决策链：`docs/RESEARCH.md` ｜ 架构决策：`docs/ADR.md` ｜ 数据模型：`docs/DESIGN.md`

## 定位

SGLang 的 KV cache 是 token 级 radix 树（`python/sglang/srt/mem_cache/radix_cache.py`），
但官方只暴露 `sglang:cache_hit_rate` 一个聚合指标。KVScope 读取**运行时快照**
（radix 树结构 dump），离线回答：

- 树长什么样？（节点数、深度分布、共享结构）
- 共享率多高？（重复 token 占比）
- 碎片多严重？（分裂节点占比）
- 驱逐效率如何？（evictable/protected、被驱逐节点位置）
- 命中热度分布？（hit_count 长尾）
- 多轮 agent 的 KV reuse 如何随轮次变化？（turns）
- 缓存事件流/驱逐时间线？（events）
- 人类思考间隙如何导致缓存衰减？（gaps）
- 预测性驱逐是否优于 LRU？（evict，CacheWise 方向）

## 快速开始

```bash
# 分析快照（text 报告）
kvscope analyze --input snapshot.jsonl

# JSON 输出（供脚本/CI 消费）
kvscope analyze --input snapshot.jsonl --output-format json

# per-turn KV reuse（多轮 agent 负载）
kvscope turns --input turns.jsonl

# KV 事件流（churn/驱逐时间线）
kvscope events --input events.jsonl

# gap 衰减预测（TraceLab 方向）
kvscope gaps --input turns.jsonl

# 驱逐策略对比（CacheWise 方向）
kvscope evict --input evict-compare.json
```

零依赖安装：`pip install -e .`（分析端不需要 SGLang；`scripts/` 模拟脚本才需要）。

## 端到端 Demo：SGLang 真实 radix cache → kvscope

用 SGLang 自己的 `RadixCache.create_simulated()`（纯 CPU，无需 GPU/模型）跑真实
多轮 agent 工作负载（共享系统提示词 + 分叉会话 + 驱逐 + 锁），导出快照/轮次/事件：

```bash
# 1. 在装有 sglang 的环境（如 WSL）里跑模拟并 dump 三种产物
python scripts/simulate_agent_workload.py /tmp/kvscope-demo

# 2. 用 kvscope 分析
kvscope analyze --input /tmp/kvscope-demo.snapshot.jsonl
kvscope turns   --input /tmp/kvscope-demo.turns.jsonl
kvscope events  --input /tmp/kvscope-demo.events.jsonl
kvscope gaps    --input /tmp/kvscope-demo.turns.jsonl
```

驱逐策略对比实验：

```bash
python scripts/simulate_evict_compare.py /tmp/evict-compare.json
kvscope evict --input /tmp/evict-compare.json
```

## 研究结论（可复现）

### 1. Agent 负载的 KV reuse 特征（TraceLab 复现）

多轮 agent 会话（共享系统提示词 + 分叉）：

```
per-turn hit ratio:
  turn   1:  94.1%   ← 共享系统提示词，几乎全命中
  turn   8: 100.0%   ← 完全命中
  turn   9:   0.0%   ← 冷会话首请求（完全 miss）
  turn  12:  75.0%   ← 冷会话内部累积前缀恢复
```

avg hit ratio 94.08%——与 TraceLab 论文"全局 prefix 命中 95.7%"一致。

### 2. Gap 衰减（TraceLab 点名方向）

```
[gap -> next-turn hit ratio]
        <10s:  93.0% (n=6)     ← 短间隙，缓存完好
      10-60s:  89.0% (n=25)
     5-30min:   0.0% (n=4)     ← 长间隙，缓存完全失效！
```

logistic 模型 `P(hit|gap)` 参数化为预测性驱逐信号 `cache_decay_probability(gap)`。

### 3. 预测性驱逐 vs LRU（CacheWise 方向）

```
metric                 LRU  predictive    delta
hit ratio             88.0%      88.0%    +0.0%
eviction rounds          59         42      -17
=> predictive reaches equal hit rate with 29% fewer eviction rounds
```

相同命中率下，gap 预测性驱逐（长间隙主动清理死前缀）**减少 29% 驱逐操作**。

## 快照格式

见 `docs/DESIGN.md`。快速示例：

```jsonl
{"type": "snapshot", "version": 1, "page_size": 1, "eviction_policy": "lru",
 "pool": {"total_tokens": 1000, "used_tokens": 600},
 "nodes": [
   {"id": 0, "parent": -1, "token_len": 0, "children": [1], "lock_ref": 1, "hit_count": 0, "evicted": false},
   {"id": 1, "parent": 0, "token_len": 100, "children": [2, 3], "lock_ref": 0, "hit_count": 8, "evicted": false}
 ]}
```

## 双机同步（办公本 ↔ 家庭 GPU 主机）

本项目是独立 Git 仓库（GitHub: `unravel1020/kvscope`），双机通过 GitHub 同步：

```bash
# 家庭主机首次拉取
git clone https://github.com/unravel1020/kvscope.git

# 之后同步（办公本推、家庭机拉，或反之）
git pull origin master
```

- **办公本（无 GPU）**：白天做分析端开发 + `create_simulated` 模拟验证（纯 CPU）
- **家庭主机（RTX 3080）**：晚上跑真实 SGLang 服务器 → 真实 dump → 分析（v0.7 规划）
- 研究文档（RESEARCH/ADR/DESIGN）随仓库同步，两机共享

## 开发状态

- [x] 数据模型设计（docs/DESIGN.md）
- [x] vLLM 参考实现分析
- [x] 快照解析 + 树分析（v0.1，analyze）
- [x] 端到端：SGLang 真实 radix 模拟 → dump → analyze（v0.2）
- [x] per-turn KV-reuse 分析（v0.3，turns）
- [x] KV 事件流分析（v0.4，events）
- [x] gap 衰减预测（v0.5，gaps，TraceLab 方向）
- [x] 驱逐策略对比（v0.6，evict，CacheWise 方向）
- [x] 文档：RESEARCH / ADR / DESIGN
- [ ] 真实 SGLang server dump hook（v0.7，需运行中的服务器/GPU，家庭主机）
- [ ] 跨框架对比（vLLM 数据源）
- [ ] 上游 PR（SGLang #27701 kv_cache_usage_perc / #36678 per-request metrics）

## 许可

Apache-2.0（与 SGLang/vLLM 一致）
