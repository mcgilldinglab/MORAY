# Classifier assets

Set `MBC_ASSETS_ROOT` to the MORAY classifier bundle's `assets` directory and `MBC_LABEL_SPACE` to its `config/label_space` directory. Use the databases, model weights and label tables from the same build.

| Path under `MBC_ASSETS_ROOT` | Contents |
|---|---|
| `01_L1_routing/db/domain_unique_k25_compact6_256` | Domain routing index |
| `02_fungi/db` | Fungal species 25-mer index, priors and taxonomy metadata |
| `03_bacteria/db` | Bacterial species 25-mer index, priors and taxonomy metadata |
| `04_virus/db` | Viral species, genus and phylum 25-mer indexes and priors |
| `02_fungi/model`, `03_bacteria/model`, `04_virus/model` | Frozen epoch-8 checkpoints under `epoch_checkpoints` |
| `99_shared/base_model/GENERator-v2-prokaryote-1.2b-base` | Local tokenizer configuration and pretrained backbone |

Each `MBC_LABEL_SPACE` domain directory (`bacteria`, `fungi` or `virus`) contains `metadata.tsv` and `label_space.tsv`. The classifier reads the matching paths and model settings from `config/pipeline.json` and the files in `config/deploy/`.
