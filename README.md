# pairwise-deinterlace-field-order

## 领域

游戏、图形与多媒体

## 主题

隔行场序去隔行

## 说明

场序解析、遮挡回退与切镜清矢量。

初始状态下 `media/deinterlace_field_order.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复映射

| 问题 | 入口 | 处理 |
| --- | --- | --- |
| 场序标记缺失/标反 | `parse_field_order` / `detect_field_order` | 归一化 odd/even、top/bottom、TFF/BFF、`tt/bb`；有画面时按两种场序假设的相邻场 SAD 检测真实顺序 |
| 快速横摇产生梳齿/锯齿 | `interpolate_field_pair` / `edge_directed_interpolate` | 按场序配对；运动区域只用同场相邻参考行做边缘方向内插（ELA），静态区域保留编织细节 |
| 遮挡块匹配失败后复制重复纹理 | `occlusion_fallback` / `block_match` / `fallback` | 综合 SAD、置信度和前后矢量一致性判定遮挡；拒绝候选块，回退到同场空间/边缘内插或可靠邻场 |
| 切镜后沿用旧运动矢量 | `clear_vectors_on_cut` / `detect_scene_cuts` | 硬切时清空运动矢量、双向匹配状态和场历史；支持原地清空传入的 list/ndarray/dict |
| 24p 被当成 60i 导致节奏变快 | `keep_source_rate` / `rate_to_fraction` | 保留源帧率和有理时基（含 `24000/1001`），输出时间戳按源帧率计算，不固定为场率或场率一半 |
| 结果无法回看/审计 | `review_output` / `export` / `validate_deint` / `metrics` / `audit` / `determinism` | 输出 JSON/HTML/PPM 回看工件、校验项、梳齿能量、审计事件和逐帧哈希确定性检查 |

契约测试中的整数 `1` 会进入内置合成横摇/切镜/遮挡案例，跑的是同一套处理逻辑。`tests/test_deinterlace_fixes.py` 补充了六条修复目标的行为回归。
