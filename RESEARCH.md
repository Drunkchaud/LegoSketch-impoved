# LegoSketch 改进研究

本目录是独立研究项目，原 `LegoSketch_ICML` 目录继续作为已有基线保留。
当前尚未加入新模型架构。研究起点为现有本地源码，包含此前的复现修复与真实数据评估入口，并非未经修改的官方代码。

## 来源与当前状态

- 上游：https://github.com/FFY0/LegoSketch_ICML
- 上游起点：`6e5bda6a70bcdfc6fb5190d18166cf3b10f111cc`
- `README.upstream.md` 保留作者说明；`BASELINE_PROVENANCE.json` 记录复制时各文件的 SHA256。
- 保留官方 seed0/seed1 小型预训练模型，模型格式为可信来源的 PyTorch 完整对象。
- 保留已有 CuPy 延迟导入、Cluster 清理、诊断及自定义 NPZ 评估支持。
- 训练入口仍保留本地调试设置：100 步、Item 数 100～1000、较小批量。这不是论文训练配置，正式训练前需另行明确配置。
- 不包含真实数据、合成数据、历史实验输出或训练日志。现有数据继续使用项目外的 `D:/科研/数据集/`。
- 源项目未发现 LICENSE 文件。本项目保留作者来源说明，不擅自为上游代码新增授权声明。

## 固定权重评估

在 WSL 的本项目根目录、原 Lego 环境中执行：

```bash
python EvalModel/EvalDataset/Test_Real.py \
  --npz "/mnt/d/科研/数据集/processed/kosarak.npz" \
  --budgets-kib 300 400 500 600 700 800 \
  --save-predictions --run
```

不加 `--run` 只做文件和格式检查。Webdocs 独立测试使用 `ratio_split/webdocs_freqs_test.npz`。不包含合成数据，原合成评估入口若要运行，需从基线项目复制所需数据到被忽略的 Datasets 目录。

## 每版研究的记录方式

1. 提出假设，记录修改动机与预期结果。
2. 修改代码后创建提交；记录实际评估使用的提交编号、权重来源和数据哈希。
3. 使用与基线相同的数据、内存预算、哈希及计时口径完成对照。
4. 在 `experiments/` 保存简短实验说明和汇总，不提交大体积预测、数据或临时文件。
5. 提交源码及汇总后推送 GitHub；验证过的版本可打标签。提交本身不代表效果已验证。

初始标签 `baseline-local-v0` 仅标识复制后的研究起点，后续修改在本项目进行。
