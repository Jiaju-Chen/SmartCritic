# SearchQA 远程检索器验收与 8 卡训练

这个目录隔离同学提供的 SearchQA 检索服务，不修改当前使用本地检索器的
7 卡正式实验。远程服务通过 SSH 转发到 8 卡服务器的
`127.0.0.1:18003`。

## 稳定性验收

先启动可自动重连的隧道：

```bash
bash experiments/luna_unified_critic_searchqa_remote_8card_20260829/tunnel_supervisor.sh
```

再运行默认 6 小时的稳定性测试：

```bash
python experiments/luna_unified_critic_searchqa_remote_8card_20260829/soak_remote_retriever.py \
  --output /home/dataset-local/cjj/RL/runs/luna_unified_searchqa/remote_retriever_8card_20260829/soak/records.jsonl
```

测试每 30 秒执行固定检索并核对 top-3 文档编号，每 5 分钟执行一次 32 并发
突发请求。任何请求失败、超时或结果不一致都会使最终验收失败。

## 8 卡正式训练

验收通过后使用：

```bash
bash experiments/luna_unified_critic_searchqa_remote_8card_20260829/run_formal_luna_unified_8gpu.sh
```

该入口恢复 `8 GPU`、训练任务批量 `256`、每个任务 5 条轨迹、策略和评论器
小批量 `512`，其余算法、模型、奖励、验证、保存和 200 步设置与原 SearchQA
Luna Unified 正式配置一致。检索错误会直接终止训练。
