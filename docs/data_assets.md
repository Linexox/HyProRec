# LAB data assets

The refactor worktree consumes two separately prepared semantic assets:

| Dataset | LAB asset | Items | Source snapshot | Edges |
| --- | --- | ---: | --- | --- |
| ReDial | `/data2/lhf/.DATA/hyprorec-redial` | 6,924 | `/data2/lhf/HyProRec/core/data/hocrs2_redial` | four cosine-neighbor tables, top-k 10 |
| TG-ReDial | `/data2/lhf/.DATA/hyprorec-tg-redial` | 33,834 | `/data2/lhf/.DATA/lhf-tg-redial` | four cosine-neighbor tables, top-k 10 |

Each asset contains the split JSON files, `movies_info.csv`, four offline
embedding tables, `hyperedge_table.json`, and `provenance.json`. The worktree
links them as `data/hyprorec-redial` and `data/hyprorec-tg-redial`.

The raw-modality contract differs by dataset. ReDial has no `mm/` directory in
the prepared asset, so Grounding reads its offline `img`, `vdo`, and `ado`
embeddings through a learned `LayerNorm -> Linear` feature-projection source
encoder; its text view still uses the lightweight MPNet source encoder. This is
an intentional fallback and does not change the hypergraph inputs. TG-ReDial's
`mm/` directory remains a symlink to the existing source asset and contains
complete image, video, and audio blocks, so Grounding uses the corresponding
lightweight raw source encoders for those views. CRS consumes the resulting
hypergraph towers and does not require the raw blocks at recommendation time.

The old ReDial table recorded co-occurrence metadata and the old TG-ReDial
table had no provenance. Neither is used by the refactor configurations. The
preparation entry points are linked under each asset's `scripts/` directory:
`prepare_dataset.py`, `prepare_hyperedge_table.py`, and
`prepare_embeddings.py`.
