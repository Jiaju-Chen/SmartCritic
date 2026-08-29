# SearchQA 本地检索服务验收

这个目录把 SearchQA 的检索后端部署到 8 卡服务器本机，去掉训练过程对
SSH 隧道和外部 1 卡服务器的依赖。

## 运行拓扑

```text
GPU 0-6: PPO 训练或评测
GPU 7:   官方 E5 查询编码器和官方平面内积索引
CPU:     官方 FAISS 索引文件读取和语料读取
训练环境: http://127.0.0.1:18002/retrieve
```

检索器只绑定 `127.0.0.1`，不会对外暴露 HTTP 服务。当前服务器安装的是
CPU 版 `faiss`，因此脚本读取官方 FAISS 平面索引的向量，并使用 `torch` 在
GPU 7 上执行相同的平面内积检索；编码器、索引、语料和 `/retrieve` 数据格式
均保持官方版本，吞吐量仍由压力测试确认。当前脚本使用已经复制到
8 卡服务器的官方资产：

```text
/home/dataset-local/cjj/RL/data/searchR1_official_retriever
```

## 运行验收

```bash
bash experiments/luna_unified_critic_searchqa_local_8card_20260829/start_local_retriever.sh
bash experiments/luna_unified_critic_searchqa_local_8card_20260829/acceptance_test.sh
```

验收包含健康接口、正确的单查询请求、顺序与并发延迟，以及一次真实的
SearchQA 环境搜索和最终答案奖励测试。验收失败时脚本返回非零状态，不会
把错误检索结果当成通过。

训练链路还提供一个单步冒烟脚本。它要求本地检索器已经启动，并固定让检索器
占用 GPU 7、训练使用 GPU 0--6：

```bash
bash experiments/luna_unified_critic_searchqa_local_8card_20260829/run_training_smoke_local.sh
```

这个脚本只读取官方 SearchQA 处理后数据，执行 1 个训练更新和 1 次小验证，
默认关闭在线实验记录；它不是正式训练入口。

通过验收后，正式 Luna Unified 训练使用独立入口：

```bash
bash experiments/luna_unified_critic_searchqa_local_8card_20260829/run_formal_luna_unified_7gpu.sh
```

该入口保留 2 层共享评论器主干、轮次和词元两个价值头及残差混合优势。相对
原 8 卡配置，它只把训练批量从 256 调整为 252、策略和评论器小批量从 512
调整为 504，并把训练设备数调整为 7；每个任务仍采样 5 条轨迹，其他算法、
环境、奖励、学习率和 200 步训练设置不变。检索错误会直接终止训练，避免把
通信错误作为环境观察写入轨迹。

## 训练公平性

检索器占用第 8 张卡后，正式训练应使用 GPU 0-6，并显式设置
`trainer.n_gpus_per_node=7`。这与原来的 8 卡训练不是同一个硬件预算，不能
直接把两者的速度作为公平结论。若必须保留 8 张训练卡，应把检索器放到独立
的计算节点，或者使用已经通过压力测试的集群检索服务。

正式训练每 50 步做一次完整验证并更新 `best` 和 `latest` 两个 checkpoint
槽位；WandB 使用项目 `verl_agent_searchqa_critic_ablation`。
