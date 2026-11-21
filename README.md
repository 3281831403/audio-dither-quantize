# audio-dither-quantize

## 题面

从 24 bit 降到 16 bit 导出后，静音段出现嘶嘶声、响段还发闷：抖动幅度写成 2 LSB，底噪被抬高；噪声用 random 生成，同一 seed 无法复现；量化直接截断而不是四舍五入，整体偏低半 LSB；超幅样本按回绕处理，峰值变成反相；反量化忘记除 2**(bits-1)，幅度整体错；误差反馈系数大于 1，噪声整形发散；SNR 只算功率比没有取 log，数值不对；静音段 SNR 仍做除法，直接除零；批量处理共用一条噪声序列，轨道之间互相串；bit_depth < 1 不校验，缩放时除零。dither/quantize.py 是核心，tests/test_quantize.py 15个红。要修的是：把抖动幅度改回 1 LSB 并用显式 seed 的确定性哈希生成噪声、把量化改成四舍五入并饱和钳制、反量化除以 2**(bits-1)、把整形系数限制在稳定范围、把 SNR 改为 dB 且静音段返回 inf、为批量每轨派生独立 seed、校验 bit_depth >= 1，并在任一轨非法时整体回滚。

## 运行方式

```powershell
cd D:\gsb-20261119\audio-dither-quantize
python -m pytest tests -q
```

初始失败环境应恰好输出 `15 failed`，没有 collection/import/syntax error。15 个测试全部直接调用 `dither/quantize.py` 里的桩函数，修复实现后应全部转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/` 不可修改。实现只使用标准库，离线、无随机、不 sleep；噪声由显式 `seed` 经确定性哈希生成。音频统一为归一化 `list[float]`，样本范围 `-1..1`，量化误差容差为 `1 LSB`；参数或数据非法时抛出 `DitherError`。
