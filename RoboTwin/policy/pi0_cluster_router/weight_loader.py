"""Load the frozen pi0 base and the seven-expert LoRA bank into one model."""

from dataclasses import dataclass
from pathlib import Path

import flax.traverse_util
import jax
import numpy as np


@dataclass(frozen=True)
class RoutedLoRABankWeightLoader:
    base_params_path: str
    bank_path: str

    def load(self, params):
        from openpi.models import model as model_lib
        from openpi.shared import download

        reference = flax.traverse_util.flatten_dict(params, sep="/")
        base = model_lib.restore_params(
            download.maybe_download(self.base_params_path), restore_type=np.ndarray)
        base = flax.traverse_util.flatten_dict(base, sep="/")

        # The released pi0 checkpoint supplies every original parameter.  New
        # router/residual/bank leaves retain their model initialization here.
        merged = {}
        missing_original = []
        for key, value in reference.items():
            if key in base:
                merged[key] = np.asarray(base[key], dtype=value.dtype)
            else:
                merged[key] = value
                if "lora" not in key:
                    missing_original.append(key)
        if missing_original:
            raise ValueError(
                "Base checkpoint is missing non-LoRA model parameters: "
                + ", ".join(missing_original[:8]))

        bank_file = Path(self.bank_path)
        if not bank_file.is_file():
            raise FileNotFoundError(f"Missing routed LoRA bank: {bank_file}; run prepare first")
        with np.load(bank_file, allow_pickle=False) as bank:
            expected_bank = {key for key in reference if "lora_bank_" in key}
            if set(bank.files) != expected_bank:
                missing = sorted(expected_bank - set(bank.files))
                extra = sorted(set(bank.files) - expected_bank)
                raise ValueError(f"LoRA bank keys differ from model: missing={missing}, extra={extra}")
            for key in bank.files:
                value = np.asarray(bank[key])
                target = reference[key]
                if value.shape != target.shape:
                    raise ValueError(
                        f"LoRA bank shape mismatch for {key}: {value.shape} != {target.shape}")
                merged[key] = value.astype(target.dtype, copy=False)

        result = flax.traverse_util.unflatten_dict(merged, sep="/")
        # Keep shape validation local so failures happen before GPU allocation.
        jax.tree.map(lambda expected, got: _check_shape(expected, got), params, result)
        return result


def _check_shape(expected, got):
    if expected.shape != got.shape or expected.dtype != got.dtype:
        raise ValueError(
            f"Loaded parameter differs from reference: {got.shape}/{got.dtype} "
            f"!= {expected.shape}/{expected.dtype}")
    return got
