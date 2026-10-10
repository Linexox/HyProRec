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
