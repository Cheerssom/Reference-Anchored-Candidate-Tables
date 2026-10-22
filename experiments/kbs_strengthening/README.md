# 本地实验：暂停、重启与低资源运行

在项目根目录的 PowerShell 中执行。默认使用本机已有的
`python`，可通过 `-Python` 指定其他已配置环境。

```powershell
# 查看已完成阶段
.\scripts\run_kbs_strengthening.ps1 -Stage Status

# 从有效断点继续完整流程
.\scripts\run_kbs_strengthening.ps1 -Stage Resume

# 请求在安全点暂停；也可以在正在运行的终端按 Ctrl+C
.\scripts\run_kbs_strengthening.ps1 -Stage Pause
```

实验按下载、冻结特征、CIFAR、Pets、ImageNet-R 顺序串行运行。同一时间
只允许一个本项目 GPU 阶段，系统会在进程退出后自动释放互斥锁。
不要删除 `gpu.lock` 来绕过正在运行的任务。

默认 CPU 两线程；PyTorch 显存分配上限为显卡容量的 75%，在本机 8GB
显卡上约为 6GB，驱动和桌面还会占用额外显存。图像编码批次为 16，
遇到 CUDA 显存不足会减半，最小为 1。进一步降低负载可在启动前设置：

```powershell
$env:KBS_CPU_THREADS = '1'
$env:KBS_GPU_FRACTION = '.60'
.\scripts\run_kbs_strengthening.ps1 -Stage Resume
```

下载按范围分片保存。每批图像特征单独保存；模型按类保存；距离计算按类
保存；实验结果按类别划分、到达顺序和阶段保存。写入使用临时文件及原子
替换，运行中断后只重做最后一个未完成单元。特征续跑会校验数据清单哈希，
实验续跑会校验代码、协议和缓存哈希；版本变化不会悄悄复用旧结果。

`logs/` 存放运行日志；`resume_test.json` 记录真实暂停后恢复的检验。
`cifar/p*/complete.json` 等文件表示相应划分已完成。
最终比较必须等待三个划分全部完成，不能把中间记录当作论文最终结果。

最终 FeNeC 对照在各数据集的 `fenec_vit/` 中，使用官方 ViT 单次收缩设置。此前双次收缩探针保留在上层目录。论文生成器会明确读取更正后的版本，详细原因见 `experiments/kbs_fenec_vit_correction.md`。

## 本轮稳健性与稠密缓存实验

```powershell
# 仅续跑新增实验，不重复图像编码
.\scripts\run_kbs_strengthening.ps1 -Stage Depth
```

结果保存在 `experiments/kbs_depth/`，含 9 个稠密缓存划分和 81 个支持预算/校准配置；`Status` 显示完成数（90/90）。沿用同一个 PAUSE 标志和 GPU 互斥锁。真实暂停恢复记录见 `experiments/kbs_depth/resume_validation.json`：已完成文件哈希与修改时间不变。

验证与表格重建：

```powershell
python src/experiments/test_kbs_depth.py
python src/experiments/kbs_verify_depth.py
python src/experiments/kbs_depth_cost.py
python paper/kbs/build_depth_results.py
```

冻结特征缺失时先按原流程生成缓存。81 个配置是相关子实验，论文按三个类别划分汇总，不是81个独立随机种子。新增实验复用本地环境与数据，无付费计算。

## 旧类验证查询量实验（CPU）

```powershell
.\scripts\run_kbs_strengthening.ps1 -Stage ValidationBudget
python src/experiments/test_kbs_validation_budget.py
python src/experiments/kbs_verify_validation_budget.py
python paper/kbs/build_validation_budget_results.py
```

本阶段只读 `experiments/kbs_depth/` 的预测档案，结果保存在 `experiments/kbs_validation_budget/`。不重新训练或编码图像，也不重新选择融合参数。使用同一 PAUSE 标志，每个会话原子保存；续跑核对源代码、协议和原始输入哈希。`Status` 显示三个数据集完成状态。

## 十类别划分与第二骨干（当前主实验）

```powershell
.\scripts\run_kbs_strengthening.ps1 -Stage Revision
python src/experiments/test_kbs_revision.py
python src/experiments/kbs_verify_revision.py
python paper/kbs/build_revision_results.py
```

使用现有环境和本地 OpenAI CLIP B/32 权重；不安装新环境、不使用付费计算。数据池与先前实验一致。`experiments/kbs_revision/b32_features/` 下每16张图像保存一个特征块，`l14/` 和 `b32/` 下按类别划分和会话保存指标。共享 PAUSE/Resume 与 GPU 互斥锁继续适用。Status 显示60个划分运行的完成数。

当前主实验为每数据集十个不同的类别划分；原来的三划分在其中。置信区间按配对类别划分均值重采样，不把到达顺序、会话或早先81个相关配置当成独立数据集。
