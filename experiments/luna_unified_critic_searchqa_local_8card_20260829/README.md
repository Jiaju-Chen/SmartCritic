# SearchQA 本地检索服务验收

这个目录把 SearchQA 的检索后端部署到 8 卡服务器本机，去掉训练过程对
SSH 隧道和外部 1 卡服务器的依赖。

## 运行拓扑

```text
GPU 0-6: PPO 训练或评测
GPU 7:   官方 E5 检索器和 FAISS GPU 索引
训练环境: http://127.0.0.1:18002/retrieve
```

检索器只绑定 `127.0.0.1`，不会对外暴露 HTTP 服务。当前脚本使用已经复制到
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

## 训练公平性

检索器占用第 8 张卡后，正式训练应使用 GPU 0-6，并显式设置
`trainer.n_gpus_per_node=7`。这与原来的 8 卡训练不是同一个硬件预算，不能
直接把两者的速度作为公平结论。若必须保留 8 张训练卡，应把检索器放到独立
的计算节点，或者使用已经通过压力测试的集群检索服务。

本目录只负责本地检索器和验收，不会自动启动大规模训练。
