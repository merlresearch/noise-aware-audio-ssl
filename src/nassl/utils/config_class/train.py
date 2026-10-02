# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class DMConfig(BaseModel):
    dataloader: Dict[str, Any]
    dataset: Dict[str, Any]
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: Dict[str, Any]


class DMSplitConfig(BaseModel):
    train: DMConfig
    valid: Optional[DMConfig] = None


class CallbackConfig(BaseModel):
    tqdm_refresh_rate: int = 1
    callbacks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    other_callbacks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    seed: int
    name: str
    result_dir: Path

    plmodel: Dict[str, Any]
    trainer: Dict[str, Any]
    datamodule: DMSplitConfig

    refresh_rate: int = 1
    callback: CallbackConfig

    @model_validator(mode="after")
    def validate_ssl_consistency(self) -> "TrainConfig":
        guidance_msg = "Please check the config and add the corresponding case " "in the validator if necessary."

        assert self.datamodule.valid is not None

        feature_dir = self.datamodule.valid.dataset.get("feature_dir", None)
        if feature_dir is not None:
            split_feature_dir = feature_dir.split("/")
            assert split_feature_dir[-1] == "data"
            assert split_feature_dir[-3] == "valid"

            feature_name = split_feature_dir[-2]

            if feature_name == "s_beats_iter3":
                ssl_from_feature = "beats_iter3"
            elif feature_name == "s_eat":
                ssl_from_feature = "eat"
            elif feature_name == "s_dasheng":
                ssl_from_feature = "dasheng"
            else:
                raise NotImplementedError(f"Unexpected feature_dir: {feature_dir}. {guidance_msg}")

        teacher_model_cfg = self.plmodel.get("teacher_model_cfg", None)
        if teacher_model_cfg is not None:
            model = teacher_model_cfg["_target_"]
            model_id = teacher_model_cfg.get("model_id", None)

            if model_id is None:
                raise ValueError(f"teacher_model_cfg must specify model_id. {guidance_msg}")

            if model == "nassl.feature_extraction_model.FE_BEATs":
                if model_id == "BEATs_iter3":
                    ssl_from_teacher = "beats_iter3"
                else:
                    raise NotImplementedError(
                        f"Unexpected teacher model ({model}) with " f"model_id ({model_id}). {guidance_msg}"
                    )
            elif model == "nassl.feature_extraction_model.FE_EAT":
                if model_id == "worstchan/EAT-base_epoch30_pretrain":
                    ssl_from_teacher = "eat"
                else:
                    raise NotImplementedError(
                        f"Unexpected teacher model ({model}) with " f"model_id ({model_id}). {guidance_msg}"
                    )
            elif model == "nassl.feature_extraction_model.FE_Dasheng":
                if model_id == "mispeech/dasheng-base":
                    ssl_from_teacher = "dasheng"
                else:
                    raise NotImplementedError(
                        f"Unexpected teacher model ({model}) with " f"model_id ({model_id}). {guidance_msg}"
                    )
            else:
                raise NotImplementedError(f"Unexpected teacher model: ({model}). {guidance_msg}")

            if feature_dir is not None and ssl_from_feature != ssl_from_teacher:
                raise ValueError(
                    "The SSL model inferred from feature_dir "
                    f"({ssl_from_feature}) is different from the one inferred "
                    f"from teacher_model_cfg ({ssl_from_teacher})."
                )

        return self

    @model_validator(mode="after")
    def validate_save_only_trainable_with_ema(self) -> "TrainConfig":
        save_only_trainable = self.plmodel.get("save_only_trainable", False)

        using_ema = len(self.callback.other_callbacks) > 0

        if using_ema and save_only_trainable:
            raise ValueError("save_only_trainable is not compatible with using " "TrainableOnlyEMACallback.")

        return self
