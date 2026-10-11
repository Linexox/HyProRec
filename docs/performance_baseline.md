# Refactor performance notes

The first measured optimization is the decoder-only path in `HoCRSBaseModel.encode`.
It computes the same final hidden state and causal logits without retaining every
decoder layer. The measurement uses LAB GPU 0, Qwen2.5-Omni-3B, batch size 1,
sequence length 96, inference mode, and three timed iterations after two warmups:

| Path | Time (ms) | Peak allocated (GiB) |
| --- | ---: | ---: |
| Full causal-LM with all hidden states | 36.59 | 8.929 |
| Decoder with final hidden state only | 37.31 | 8.897 |

The maximum absolute difference of the final hidden state and logits was `0.0`.
A train-like batch with sequence length 24 and a backward pass measured 118.70 ms
versus 112.95 ms, with peak allocation 9.005 versus 8.995 GiB. These are local
backbone measurements rather than the end-to-end acceptance result; the 30% target
still requires a fixed-data CRS comparison with the same seed, update batch, metrics,
and generation settings.

The CRS path now avoids projecting the vocabulary head at prompt and graph-token
positions when labels are present. It projects only the shifted positions whose
labels are valid, and reconstructs the teacher-forced argmax tensor with `-100`
elsewhere. GPT-2 reference tests and a Qwen2.5-Omni-3B GPU smoke matched the dense
loss and argmax results exactly. This is a correctness check; an end-to-end speed
claim still requires the fixed-data comparison above.

The data path also caches decoded histories, prompt tokenization, response
tokenization, and strict local hypergraphs. Token caches use a bounded LRU (8192
entries by default), so reuse does not retain an entire TG-ReDial epoch in host
memory. `GraphTokenMoE` evaluates its expert
weights with batched matrix multiplications. Trainer diagnostics now compute only
the SwanLab-required attention masses and per-view router means; entropy, top-k,
maximum-weight, top-1, and standard-deviation summaries are not materialized.
These changes reduce repeated Python work and logging overhead while preserving
the tensors used by the recommendation and dialogue losses. They still require
the fixed-data end-to-end benchmark before accepting the 30% throughput target.
