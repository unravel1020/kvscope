# 上游 PR 候选清单（kvscope 反哺计划）

> 目标：以「给 SGLang 补官方缺口」的 PR 建立社区声望，反哺 kvscope 项目
> （ADR-009 依据：kv_cache_usage_perc 等缺口正是 kvscope 分析的数据基础）。
> 状态：待执行（v0.6 之后），优先级从高到低。

## 为什么提上游 PR

1. **建立贡献者履历**：kvscope 是"读数据"的工具，上游补指标 = 让 kvscope 能读到更多
2. **社区声望**：SGLang maintainer 认识你后，kvscope 的曝光和可信度都提升
3. **学习闭环**：改上游代码 = 深度理解 mem_cache 数据流

## 候选 1：kv_cache_usage_perc（对标 vLLM）

- **Issue**：sgl-project/sglang #5979（2024 年提出至今 open，"对标 vLLM 的
  vllm:gpu_cache_usage_perc"）
- **PR**：#27701（有人实现过但未合并，可参考其 diff）
- **改动**：`metrics_collector.py` 加一个 gauge——改动最小，适合第一个上游 PR
- **风险**：已有 PR 未合并说明审查慢，需持续推进

## 候选 2：OpenAI 兼容 API per-request metrics

- **Issue**：sgl-project/sglang #36678（open，请求方要 queue/prefill/decode/TTFT/ITL 分解）
- **参考**：vLLM 已有 per_request_metrics（PR #46768），可"照抄作业"
- **改动**：`/v1/chat/completions` 响应或 meta 里加指标——中等改动
- **与 kvscope 关系**：这正是 kvscope turns/events 分析的**真实数据源**！
  上游补上后 kvscope 可以直接读 OpenAI 兼容 API 的指标，无需自己 dump

## 候选 3：per-request radix hit 长度 API

- **现状**：SGLang 有全局 `cache_hit_rate` + `/generate` 的 `cached_tokens`，
  但无 per-request hit 长度 API（kvscope 需要自己算）
- **改动**：`meta_info` 加 `prefix_hit_length` 字段——小改动，价值高
- **与 kvscope 关系**：上游提供后 kvscope turns 分析可直接消费，无需 parent 链累加

## 候选 4：KVCacheEventRecorder 纯 Python 哈希 fallback

- **现状**：`enable_kv_cache_events=True` 依赖 HiCache native hash C++ 扩展，
  纯 CPU 环境无法使用（v0.4 踩坑）
- **改动**：给 `compute_node_hash_values` 加纯 Python fallback——让事件流在
  无 native 扩展时也能工作
- **与 kvscope 关系**：kvscope v0.4 自实现了确定性哈希，若上游提供 fallback
  则可直接复用官方格式

## 提交流程（遵循 SGLang 规范）

1. 读 `CONTRIBUTING.md` + `.github/pull_request_template.md`
2. 标题用 `[Perf]` / `[Feat]` / `[Test]` 前缀（commit 风格 `[module]` 用于提交信息）
3. 必配 unit test（CPU 可跑）；注册 `register_cpu_ci` 进 CI
4. 提 PR 后等 `run-ci` 标签（maintainer 加），**不催促**
5. 与 kvscope 联动：PR 合并后，kvscope 文档标注"数据源已上游化"

## 优先级建议

先做 **候选 1**（改动最小、需求最硬）→ 再做 **候选 3**（与 kvscope 最相关）→
候选 2（中等改动）→ 候选 4（需维护者认可方向）。
