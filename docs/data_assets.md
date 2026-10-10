# LAB data assets

The refactor worktree consumes two separately prepared semantic assets:

| Dataset | LAB asset | Items | Source snapshot | Edges |
| --- | --- | ---: | --- | --- |
| ReDial | `/data2/lhf/.DATA/hyprorec-redial` | 6,924 | `/data2/lhf/HyProRec/core/data/hocrs2_redial` | four cosine-neighbor tables, top-k 10 |
| TG-ReDial | `/data2/lhf/.DATA/hyprorec-tg-redial` | 33,834 | `/data2/lhf/.DATA/lhf-tg-redial` | four cosine-neighbor tables, top-k 10 |

Each asset contains the split JSON files, `movies_info.csv`, four offline
embedding tables, `hyperedge_table.json`, and `provenance.json`. The worktree
links them as `data/hyprorec-redial` and `data/hyprorec-tg-redial`. TG-ReDial's
large raw `mm/` directory remains a symlink to the existing source asset; it is
not copied into the refactor snapshot because the current training path reads
offline embeddings.

The old ReDial table recorded co-occurrence metadata and the old TG-ReDial
table had no provenance. Neither is used by the refactor configurations. The
preparation entry points are linked under each asset's `scripts/` directory:
`prepare_dataset.py`, `prepare_hyperedge_table.py`, and
`prepare_embeddings.py`.
