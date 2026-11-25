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
