# KVScope — SGLang KV Cache 显微镜

> 离线分析 SGLang radix cache 的结构、共享率、碎片与驱逐效率。
> 数据模型与设计文档：`docs/DESIGN.md`（labs-notes/KV-SCOPE-DESIGN.md 同步）
> 参考蓝本：vLLM 官方离线 prefix-cache analyzer 的 CLI/报告模式（PR #48369）

## 定位

SGLang 的 KV cache 是 token 级 radix 树（`python/sglang/srt/mem_cache/radix_cache.py`），
但官方只暴露 `sglang:cache_hit_rate` 一个聚合指标。KVScope 读取**运行时快照**
（radix 树结构 dump），离线回答：

- 树长什么样？（节点数、深度分布、共享结构）
- 共享率多高？（重复 token 占比）
- 碎片多严重？（分裂节点占比）
- 驱逐效率如何？（evictable/protected、被驱逐节点的位置）
- 命中热度分布？（hit_count 长尾）

## 用法

```bash
# 分析快照（text 报告）
kvscope analyze --input snapshot.jsonl

# JSON 输出（供脚本/CI 消费）
kvscope analyze --input snapshot.jsonl --output-format json

# 指定 top-K 共享组
kvscope analyze --input snapshot.jsonl --top-k-groups 10
```

## 快照格式

见 `docs/DESIGN.md` 第三节。快速示例：

```jsonl
{"type": "snapshot", "version": 1, "page_size": 1, "eviction_policy": "lru",
 "pool": {"total_tokens": 1000, "used_tokens": 600},
 "nodes": [
   {"id": 0, "parent": -1, "token_len": 0, "children": [1], "lock_ref": 1, "hit_count": 0, "evicted": false},
   {"id": 1, "parent": 0, "token_len": 100, "children": [2, 3], "lock_ref": 0, "hit_count": 8, "evicted": false}
 ]}
```

## 端到端 Demo（v0.2）：SGLang 真实 radix cache → kvscope

用 SGLang 自己的 `RadixCache.create_simulated()`（纯 CPU，无需 GPU/模型）跑真实的多轮
agent 工作负载（共享系统提示词 + 分叉会话 + 驱逐 + 锁），导出快照后交给 kvscope 分析：

```bash
# 1. 在装有 sglang 的环境（如 WSL）里跑模拟并 dump
python scripts/simulate_agent_workload.py /tmp/kvscope-demo.jsonl

# 2. 用 kvscope 分析真实快照
kvscope analyze --input /tmp/kvscope-demo.jsonl
```

产出（示例）：
```
[sharing]
  reuse ratio       : 25.04%      ← 512-token 系统提示词被 3 会话共享
  shared nodes      : 1
[fragmentation]
  small nodes       : 3           ← 分裂残余（32/8/3 token）
[eviction]
  locked nodes      : 1 (512 tokens)
[hotness]
  top hit nodes      : #1(10), #2(4), #6(4)
```

## 开发状态

- [x] 数据模型设计（docs/DESIGN.md）
- [x] vLLM 参考实现分析
- [x] 快照解析 + 树分析（v1）
- [x] CLI（text/json 报告）
- [x] 单测（合成快照 + dump）
- [x] 端到端：SGLang 真实 radix 模拟 → dump → analyze（v0.2）
- [ ] 真实 SGLang server dump hook（v0.3，需运行中的服务器/GPU）

## 许可

Apache-2.0（与 SGLang/vLLM 一致）
