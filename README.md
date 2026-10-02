<!--
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Noise-Aware Audio Self-Supervised Learning (NA-SSL)

NA-SSL is a framework to extract clean SSL representations from noisy input by using auxiliary noise information.
This repository contains the pretrained models and training recipes used in our
[IWAENC 2026](https://arxiv.org/abs/2607.16688) and [DCASE 2026](https://arxiv.org/abs/2608.00447) workshop submissions,
and our [DCASE 2026 Task 2 Challenge submission](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Fujimura_17_t2.pdf).

<p align="center">
 <img src="/docs/overview_na.png" alt="NA-SSL Overview" width="500" border="0" />
</p>

If you use any part of this code for your work, we ask that you include at least one of the following citations:

    @InProceedings{Fujimura2026IWAENC_nassl,
      author    =  {Fujimura, Takuya and Masuyama, Yoshiki and Wichern, Gordon and Boeddeker, Christoph and Richter, Julius and {Le Roux}, Jonathan},
      title     =  {{NABEATs}: Noise-Aware Audio Representation Learning},
      booktitle =	 {Proc. International Workshop on Acoustic Signal Enhancement (IWAENC)},
      year      =	 2026,
      month     =	 sep
    }

    @InProceedings{Fujimura2026DCASE_nassl,
      author    =  {Fujimura, Takuya and Wichern, Gordon and Masuyama, Yoshiki and Boeddeker, Christoph and Saijo, Kohei and Richter, Julius and Edo, Takahiro and {Le Roux}, Jonathan},
      title     =  {Anomalous Sound Detection Meets Noise-Aware Self-Supervised Learning},
      booktitle =	 {Proc. Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE)},
      year      =	 2026,
      month     =	 oct
    }

## Supported Pretrained Models

The following pretrained checkpoints are provided through GitHub Releases.

| Work        | Model           | Model ID                                  |
| ----------- | --------------- | ----------------------------------------- |
| IWAENC 2026 | D-BEATs         | `iwaenc2026_dbeats_seed0`                 |
| IWAENC 2026 | NA-BEATs (CA)   | `iwaenc2026_nabeats_ca_seed0`             |
| IWAENC 2026 | NA-BEATs (FiLM) | `iwaenc2026_nabeats_film_seed0`           |
| DCASE 2026  | NA-BEATs        | `dcase2026_nabeats_seed{0,1,2}`           |
| DCASE 2026  | NA-EAT          | `dcase2026_naeat_seed{0,1,2}`             |
| DCASE 2026  | NA-Dasheng      | `dcase2026_nadasheng_seed{0,1,2}`         |


## Installation

Python 3.11 or later is required.

### Pretrained model cache

Pretrained checkpoints are downloaded automatically from GitHub Releases and cached in `~/.cache/nassl`.
<details>
<summary> Set the `NASSL_CACHE_DIR` environment variable if you want to use a different cache directory</summary>

```bash
export NASSL_CACHE_DIR=/path/to/cache
```
</details>

### Inference

<details>
<summary>`nassl` package can be installed directly from GitHub</summary>

For example,

```bash
uv add git+https://github.com/merlresearch/noise-aware-audio-ssl.git
```

or

```bash
pip install git+https://github.com/merlresearch/noise-aware-audio-ssl.git
```

</details>

<details><summary>Example usage of the pretrained models</summary>

NA-BEATs example
```python
import torch

from nassl.utils.restore import restore_model

model = restore_model(model_id="dcase2026_nabeats_seed0", device="cpu")
model.eval()
x = torch.randn(1, 16000 * 10) # Input noisy (batch_size, time)
n_ref = torch.randn(1, 16000 * 10) # Reference noise (batch_size, time)
with torch.inference_mode():
    z = model(x, n_ref) # Estimate clean representation
```

D-BEATs example
```python
model = restore_model(model_id="iwaenc2026_dbeats_seed0", device="cpu")
model.eval()
x = torch.randn(1, 16000 * 10) # Input noisy only
with torch.inference_mode():
    z = model(x)
```

Locally stored checkpoints can be loaded by passing `ckpt_path` instead of `model_id`:

```python
model = restore_model(ckpt_path="/path/to/model.ckpt", device="cpu")
```

</details>



### Training

<details><summary>1. Clone the repository and install the training dependencies</summary>


```bash
git clone https://github.com/merlresearch/noise-aware-audio-ssl.git
cd noise-aware-audio-ssl
uv sync --extra train --locked
```

</details>



<details><summary>2. Download the Datasets</summary>

The training recipes use **FSD50K** as the target-sound dataset and **WHAM!**, **DEMAND**, and **QUT-NOISE** as noise datasets.

Run the provided download scripts:

```bash
bash jobs/download/fsd50k.sh
bash jobs/download/wham.sh
bash jobs/download/demand.sh
bash jobs/download/qut_noise.sh
```

</details>

<details><summary>3. Run the Common Preprocessing</summary>

Preprocess the base training datasets using:

```bash
bash jobs/preprocessing/base_training_data.sh
```

</details>


<details><summary>4. Recipes for IWAENC 2026</summary>

By default, `jobs/train/iwaenc_2026.sh` is configured to use two GPUs.
The training scripts iterate over the experiment configurations.

```bash
bash jobs/preprocessing/iwaenc_2026.sh
bash jobs/train/iwaenc_2026.sh
```


</details>

<details><summary>5. Recipes for DCASE 2026</summary>

By default, `jobs/train/dcase_2026.sh` is configured to use four GPUs.
The training scripts iterate over the experiment configurations and random seeds.

```bash
bash jobs/preprocessing/dcase_2026.sh
bash jobs/train/dcase_2026.sh
```


</details>

By default, training results are saved in `noise-aware-audio-ssl/scratch/result/`.
Pretrained checkpoints can be loaded using `model = restore_model(ckpt_path="/path/to/model.ckpt")`, as described in the [inference](#inference) section.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and license

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```
### BEATs

The following files:

* `src/nassl/model/beats/backbone.py`
* `src/nassl/model/beats/BEATs.py`
* `src/nassl/model/beats/modules.py`
* `src/nassl/model/beats/quantizer.py`
* `src/nassl/model/beats/Tokenizers.py`

were taken from BEATs with only minor modifications to import statements.
```
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2022 Microsoft
```

The following files:

* `src/nassl/model/beats_extension/dbeats.py`
* `src/nassl/model/beats_extension/nabeats.py`

were adapted from BEATs:

* Code source: [BEATs](https://github.com/microsoft/unilm/tree/master/beats)
* License: MIT ([local copy](LICENSES/MIT.txt), [upstream license](https://github.com/microsoft/unilm/blob/master/LICENSE))
  * Includes FAIRSEQ
    * License: MIT ([local copy](LICENSES/MIT.txt), [upstream license](https://github.com/facebookresearch/fairseq/blob/main/LICENSE))
  * Includes VQGAN
    * License: MIT ([local copy](LICENSES/MIT.txt), [upstream license](https://github.com/CompVis/taming-transformers/blob/master/License.txt)).
```
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2022 Microsoft
Copyright (c) 2019 Facebook, Inc. and its affiliates.
Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
```

### EAT

The following files:

* `src/nassl/model/eat/tools.py`
* `src/nassl/model/eat_extension/naeat.py`

were adapted from EAT:

* Code source: [`worstchan/EAT-base_epoch30_pretrain`](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain)
* Original repository: [`cwx-worst-one/EAT`](https://github.com/cwx-worst-one/EAT)
* License: MIT ([local copy](LICENSES/MIT.txt), [upstream license](https://github.com/cwx-worst-one/EAT/blob/main/LICENSE))
```
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2024 Wenxi Chen

```

### Dasheng

The following file:

* `src/nassl/model/dasheng_extension/nadasheng.py`

was adapted from Dasheng:

* Code source: [`mispeech/dasheng-base`](https://huggingface.co/mispeech/dasheng-base/tree/main)
* Original repository: [`RicherMans/Dasheng`](https://github.com/RicherMans/Dasheng)
* License: Apache License 2.0 ([local copy](LICENSES/Apache-2.0.txt), [upstream license](https://github.com/RicherMans/Dasheng/blob/main/LICENSE))
```
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2024 Xiaomi Corporation

```
