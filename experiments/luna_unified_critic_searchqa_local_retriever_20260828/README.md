# Local SearchQA retriever on 8card

This experiment removes the cross-machine SSH tunnel from SearchQA. The
Wikipedia corpus, E5 model, and FAISS index live on the 8card host. The default
configuration keeps the exact flat index on CPU and places only the E5 query
encoder on GPU 7, so PPO can retain its eight-GPU world size.

Required assets:

```text
/home/dataset-local/cjj/RL/data/searchR1_official_retriever/index/e5_Flat.index
/home/dataset-local/cjj/RL/data/searchR1_official_retriever/corpus/wiki-18.jsonl
/home/dataset-local/cjj/RL/data/searchR1_official_retriever/models/e5-base-v2
```

The local service runs on port 18002. Before a formal run, benchmark sequential
and concurrent requests. If CPU FAISS is too slow, do not silently change the
training world size; record a separate seven-GPU retriever experiment instead.

Search backend failures are configured as fatal for the formal run. This keeps
connection errors from being converted into ordinary tool observations and
silently contaminating PPO trajectories.
