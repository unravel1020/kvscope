# KVScope 鈥?SGLang KV Cache 鏄惧井闀?
> 绂荤嚎鍒嗘瀽 SGLang radix cache 鐨勭粨鏋勩€佸叡浜巼銆佺鐗囦笌椹遍€愭晥鐜囥€?> 鏁版嵁妯″瀷涓庤璁℃枃妗ｏ細`docs/DESIGN.md`锛坙abs-notes/KV-SCOPE-DESIGN.md 鍚屾锛?> 鍙傝€冭摑鏈細vLLM 瀹樻柟绂荤嚎 prefix-cache analyzer 鐨?CLI/鎶ュ憡妯″紡锛圥R #48369锛?
## 瀹氫綅

SGLang 鐨?KV cache 鏄?token 绾?radix 鏍戯紙`python/sglang/srt/mem_cache/radix_cache.py`锛夛紝
浣嗗畼鏂瑰彧鏆撮湶 `sglang:cache_hit_rate` 涓€涓仛鍚堟寚鏍囥€侹VScope 璇诲彇**杩愯鏃跺揩鐓?*
锛坮adix 鏍戠粨鏋?dump锛夛紝绂荤嚎鍥炵瓟锛?
- 鏍戦暱浠€涔堟牱锛燂紙鑺傜偣鏁般€佹繁搴﹀垎甯冦€佸叡浜粨鏋勶級
- 鍏变韩鐜囧楂橈紵锛堥噸澶?token 鍗犳瘮锛?- 纰庣墖澶氫弗閲嶏紵锛堝垎瑁傝妭鐐瑰崰姣旓級
- 椹遍€愭晥鐜囧浣曪紵锛坋victable/protected銆佽椹遍€愯妭鐐圭殑浣嶇疆锛?- 鍛戒腑鐑害鍒嗗竷锛燂紙hit_count 闀垮熬锛?
## 鐢ㄦ硶

```bash
# 鍒嗘瀽蹇収锛坱ext 鎶ュ憡锛?kvscope analyze --input snapshot.jsonl

# JSON 杈撳嚭锛堜緵鑴氭湰/CI 娑堣垂锛?kvscope analyze --input snapshot.jsonl --output-format json

# 鎸囧畾 top-K 鍏变韩缁?kvscope analyze --input snapshot.jsonl --top-k-groups 10
```

## 蹇収鏍煎紡

瑙?`docs/DESIGN.md` 绗笁鑺傘€傚揩閫熺ず渚嬶細

```jsonl
{"type": "snapshot", "version": 1, "page_size": 1, "eviction_policy": "lru",
 "pool": {"total_tokens": 1000, "used_tokens": 600},
 "nodes": [
   {"id": 0, "parent": -1, "token_len": 0, "children": [1], "lock_ref": 1, "hit_count": 0, "evicted": false},
   {"id": 1, "parent": 0, "token_len": 100, "children": [2, 3], "lock_ref": 0, "hit_count": 8, "evicted": false}
 ]}
```

## 绔埌绔?Demo锛坴0.2锛夛細SGLang 鐪熷疄 radix cache 鈫?kvscope

鐢?SGLang 鑷繁鐨?`RadixCache.create_simulated()`锛堢函 CPU锛屾棤闇€ GPU/妯″瀷锛夎窇鐪熷疄鐨勫杞?agent 宸ヤ綔璐熻浇锛堝叡浜郴缁熸彁绀鸿瘝 + 鍒嗗弶浼氳瘽 + 椹遍€?+ 閿侊級锛屽鍑哄揩鐓у悗浜ょ粰 kvscope 鍒嗘瀽锛?
```bash
# 1. 鍦ㄨ鏈?sglang 鐨勭幆澧冿紙濡?WSL锛夐噷璺戞ā鎷熷苟 dump
python scripts/simulate_agent_workload.py /tmp/kvscope-demo.jsonl

# 2. 鐢?kvscope 鍒嗘瀽鐪熷疄蹇収
kvscope analyze --input /tmp/kvscope-demo.jsonl
```

浜у嚭锛堢ず渚嬶級锛?```
[sharing]
  reuse ratio       : 25.04%      鈫?512-token 绯荤粺鎻愮ず璇嶈 3 浼氳瘽鍏变韩
  shared nodes      : 1
[fragmentation]
  small nodes       : 3           鈫?鍒嗚娈嬩綑锛?2/8/3 token锛?[eviction]
  locked nodes      : 1 (512 tokens)
[hotness]
  top hit nodes      : #1(10), #2(4), #6(4)
```

## 寮€鍙戠姸鎬?
- [x] 鏁版嵁妯″瀷璁捐锛坉ocs/DESIGN.md锛?- [x] vLLM 鍙傝€冨疄鐜板垎鏋?- [x] 蹇収瑙ｆ瀽 + 鏍戝垎鏋愶紙v1锛?- [x] CLI锛坱ext/json 鎶ュ憡锛?- [x] 鍗曟祴锛堝悎鎴愬揩鐓?+ dump锛?- [x] 绔埌绔細SGLang 鐪熷疄 radix 妯℃嫙 鈫?dump 鈫?analyze锛坴0.2锛?- [ ] 鐪熷疄 SGLang server dump hook锛坴0.3锛岄渶杩愯涓殑鏈嶅姟鍣?GPU锛?
## 璁稿彲

Apache-2.0锛堜笌 SGLang/vLLM 涓€鑷达級




