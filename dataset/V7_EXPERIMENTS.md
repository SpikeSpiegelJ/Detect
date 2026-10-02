# V7 YOLO26s 对照实验

本阶段只配置低学习率微调和困难样本重采样，不添加注意力或方向卷积分支，不修改原始标签。

## PyCharm 手动运行

解释器：`D:/Anaconda3/envs/yolov8/python.exe`。工作目录为项目根目录。

1. 直接运行 `train.py`：默认 `v7_control`，输出 `runs/train/yolo26s_v7_control_ft`。
2. 完成后运行 `val.py`：默认评估该对照组在原 V7 验证集上的表现。
3. 在 `train.py` 运行配置的 Parameters 填写 `--preset v7_hardneg`，运行困难采样组。
4. 在 `val.py` 的 Parameters 填写 `--preset v7_hardneg`，评估困难采样组。
5. 复查原始基线：`val.py --preset yolo26s`。需要统一评估参数后对比。

第二组同样从原始 V7 `best.pt` 初始化，**不是**从对照组初始化。不要设置旧 `--model`、`--weights` 或 `--data` 参数。
若输出目录自动产生 `-2` 后缀，验证时必须通过 `--model` 指定实际 best.pt，程序不会自动猜测最新权重。
先在 val 上选方案，最终确认才使用 `--split test`；本阶段不运行 test。

## 公共参数

两组：1280、batch 2、MuSGD、lr0=0.00005、lrf=0.1、cos_lr=True、40 epochs、patience=15、warmup=1、
mosaic=0、scale=0.15、translate=0.03、seed=0、resume=False。其余损失权重与颜色增强沿用原 YOLO26s 配置。
这是重新启动优化器的微调，不是断点续训。原权重不覆盖。

注意：相同 epoch 数不等于相同计算预算，重采样组每轮迭代更多；分析时需同时记录实际轮数与优化器更新次数。
小幅提升应换相同的另一 seed 重复两组，不能把更多训练步骤的收益全部归因于采样。

## 困难样本来源

复用 `dataset/build_focus_hardneg_manifest.py`，默认挖掘 V7 train，聚焦 class 3、6。
以候选置信度 0.001 推理；同类 IoU>=0.5、置信度>=0.25、一对一匹配成功的目标不重复。
有未匹配重点目标的图片进入困难正样本清单；同类 IoU<0.1、置信度>=0.25 的预测作为困难负样本线索，每类最多150张。
依据现有标签处理，不把含其他类别的图片改成空标签。所有原训练图片保留一次，选中图片仅追加一次，不叠加次数。

生成物：V7 根目录 `train_feeder_hardneg.txt`、`train_feeder_hardneg_audit.json`，及 `dataset/data_v7_feeder_hardneg.yaml`。
原图与标签不复制、不裁剪、不更改；val/test 路径固定。脚本拒绝覆盖现有清单。
训练集上的挖掘可能低估未见场景难度；它不是泛化评估。

本次已完成挖掘：2306张原始图片，追加338条，共2644条；274张困难正样本、71张困难负样本，其中7张重合。
有效实例 Feeder_RRU 从679增至863，Feeder_antenna 从401增至523；伴随类别 RRU 从966增至1212，
antenna_b 从1035增至1213。上述是采样曝光次数，不是新增独立图片或标注。

已通过4项离线单元测试、清单实际加载检查，以及1280/batch2的AMP前向、反向、MuSGD更新检查。
AMP初始scale=65536出现溢出，自动降低到16384后梯度有限且更新成功；未因此修改框架的缩放逻辑。
短时检查未保存模型权重，不代表完成正式训练或保证所有后续批次的稳定性。

## 验收

比较整体和两个馈线类别的 mAP50、mAP50-95，并在相近误检数量或 Precision 下比较 Recall。
观察 cut、RRU、antenna_b 等类别是否退步。不要只用降低置信度后的 Recall 判定模型识别能力提升。
若前两组无稳定收益，再单独设计方向卷积分支对照；本阶段未实现该结构实验。
