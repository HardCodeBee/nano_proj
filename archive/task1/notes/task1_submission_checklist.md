# Task 1 Final Submission Checklist

这份清单只针对当前仓库里的 `Task 1` 材料做最后一次提交视角检查。

## 1. 先区分两类“提交物”

课程里实际上有两类不同交付物：

- 报告：`Task 1 + Task 2` 写在同一份报告里
- Hugging Face 模型目录：最终用于评测的是 `Task 3` 的最佳 checkpoint，不是单纯的 `Task 1` baseline

因此：

- `Task 1` 现在最主要的作用，是把报告里的 baseline 部分写扎实
- 如果只是整理 `Task 1`，重点应放在“报告材料”和“可复现代码”上
- 真正上传到 Hugging Face 的最终模型目录，之后应以 `Task 2/Task 3` 的最佳模型为准

## 2. 当前仓库里，Task 1 已经具备的核心材料

建议作为 `Task 1` 正式材料引用的文件：

- `data/rocstories/prepare.py`
- `data/rocstories/dataset_stats.json`
- `config/train_rocstories.py`
- `out-rocstories/sample_params.json`
- `out-rocstories/task1_summary.md`
- `out-rocstories/task1_optimization_update.md`
- `README.md` 中的 ROCStories Task 1 入口说明

这些文件分别覆盖了：

- 数据处理方法
- 长度统计与 block size 选择依据
- 正式训练超参数
- 采样参数
- 最终结果摘要
- 调参过程与改进轨迹

## 3. 当前检查结论

本次检查确认：

- `Task 1` 的数据处理、训练配置、评估路径和样例生成路径已经可以自洽复现
- `prepare.py` 现在会稳定生成 `train.bin`、`val.bin`、`dataset_stats.json` 和 `test_full.txt`
- 当前文档已经明确区分“版本控制内的说明文件”和“本地生成但 gitignore 的实验产物”
- 当前最佳公开测试结果仍然是历史运行 `out-rocstories-remote-r19/`
- 该结果对应 `avg_loss = 3.216`、`ppl = 24.93`

仍然需要你手动完成的部分：

- 最终课程报告本身目前不在仓库里
- Hugging Face 最终上传目录也还没有单独清理成“只含提交必需文件”的干净文件夹

## 4. 报告怎么写

根据课程说明，报告需要注意这些硬性要求：

- 主体不超过 `2` 页
- 参考文献不计入 2 页
- Appendix 可额外放样例，但不要把关键结果只放 Appendix
- 报告应同时总结 `Task 1` 和 `Task 2`
- 如果有用 GPT 帮忙润色，必须声明，否则有扣分风险

`Task 1` 部分建议最少覆盖以下内容：

1. 数据处理
   - 数据集：`mintujupally/ROCStories`
   - tokenizer：GPT-2 BPE
   - 每条 story 末尾追加 `eot`
   - 训练使用 `train` split，本地验证使用公开 `test` split
   - 说明为什么这样做符合课程允许范围

2. 训练设置
   - 模型结构：`n_layer=6`, `n_head=6`, `n_embd=384`
   - 参数量：`29.94M`
   - `block_size=96`, `batch_size=80`
   - `learning_rate=3.5e-4`, cosine decay, `max_iters=12000`
   - `dropout=0.14`, `weight_decay=0.07`, `seed=2027`
   - 训练环境：单张 `RTX 4060 Laptop GPU`

3. 定量结果
   - 在公开测试集上的 `avg_loss = 3.216`
   - `ppl = 24.93`
   - 可以补一句：这是当前 workspace 内记录到的最佳 `Task 1` 公开测试结果

4. 定性样例
   - 放 1 到 2 组 prompt/sample 即可
   - 主文只放代表性样例，更多样例可进 Appendix

5. 简短误差分析
   - 重复
   - 结尾生硬
   - 高温采样下逻辑跳跃和语法波动更明显

建议写法：

- `Task 1` 不需要写得很花，只要清楚、完整、可复现
- 重点是把 baseline 跑通，并把超参数和结果交代清楚
- 不要把 `Task 2` 的改进点误写成 `Task 1` 的基础设定

## 5. 代码仓库怎么交

如果老师看的是代码仓库而不是你的本地工作目录，建议把“应引用的核心文件”控制在下面这批：

- `README.md`
- `data/rocstories/prepare.py`
- `data/rocstories/dataset_stats.json`
- `config/train_rocstories.py`
- `eval.py`
- `sample.py`
- `model.py`
- `out-rocstories/sample_params.json`
- `out-rocstories/task1_summary.md`
- `out-rocstories/task1_optimization_update.md`

这些文件足够支撑：

- 代码如何复现
- 数据如何处理
- 模型如何训练
- 结果如何评估
- 你最终报告里的数字来自哪里

## 6. 哪些文件建议“保留但不要当成最终提交重点”

当前仓库里还有不少历史实验文件，它们可以留在工作仓库里，但不建议在最终说明里作为主入口强调：

- `config/train_rocstories_task1_push_*.py`
- `out-rocstories-remote-r*/eval_*.log`
- `out-rocstories/task1_detailed_process.txt`

原因不是它们有问题，而是：

- 数量多
- 信息偏实验过程
- 容易分散助教对“正式基线配置”和“最终结果”的注意力

## 7. Hugging Face 模型目录怎么交

按课程说明，HF 模型目录需要非常克制。

最低要求：

- `ckpt.pt`

可选保留：

- `sample_params.json`

只有在你改了架构时才需要额外放进去的文件：

- `model.py`
- 其他自定义模型依赖文件

不建议放进 HF 模型目录的文件：

- `train.log`
- `eval_test_full.log`
- 中间 checkpoint
- 大量实验记录
- 非必要样例文件

对当前 `Task 1` 来说，如果只是做一次“基线 sanity check 的上传目录”，最干净的目录结构应类似：

```text
task1-submit/
  ckpt.pt
  sample_params.json
```

但再次提醒：

- 课程最终真正要上传评测的，是 `Task 3` 的最佳模型目录
- 不应把 `Task 1` baseline 误当成最终提交 checkpoint

## 8. 最后提交前的人工核对

提交报告前，逐条确认：

- `Task 1` 写的是 baseline，不混入 `Task 2` 的探索性改动
- 所有数字和仓库中的 `task1_summary.md` 一致
- 参数量明确写出且没有超过 `32M`
- 样例是你当前模型真实生成的，不是手写润色
- 如果用了 LLM 帮忙润色报告，已显式声明

提交 HF 模型前，逐条确认：

- 目录里只有最终 checkpoint
- 没有中间 checkpoint
- 没有无关日志
- 如果用了自定义结构，相关 `model.py` 已包含
- 目录可以被 `hf_load.py` 直接上传

## 9. 当前最稳妥的提交口径

如果现在就要写 `Task 1` 部分，建议统一使用下面这套口径：

- 数据准备脚本：`data/rocstories/prepare.py`
- 正式训练配置：`config/train_rocstories.py`
- 当前最佳公开测试结果：`avg_loss = 3.216`, `ppl = 24.93`
- 结果摘要：`out-rocstories/task1_summary.md`
- 调参过程：`out-rocstories/task1_optimization_update.md`
- 推荐采样参数：`out-rocstories/sample_params.json`

这样最稳，最不容易在报告、代码和后续提交之间说法打架。
