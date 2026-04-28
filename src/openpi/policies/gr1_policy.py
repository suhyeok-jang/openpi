import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# GR-1 (GR1ArmsAndWaistFourierHands) raw state/action is 44-dim. We train on the
# 29-dim subset used by GR00T's fourier_gr1_arms_waist modality:
#   left_arm  [0:7]   (7)
#   left_hand [7:13]  (6)
#   right_arm [22:29] (7)
#   right_hand[29:35] (6)
#   waist     [41:44] (3)
# Excluded: left_leg[13:19], neck[19:22], right_leg[35:41].
GR1_ACTIVE_DIM = 29


def _slice_gr1_29(x: np.ndarray) -> np.ndarray:
    """Slice raw 44-dim GR-1 state/action to the 29-dim arms+hands+waist subset."""
    x = np.asarray(x)
    return np.concatenate([x[..., 0:13], x[..., 22:35], x[..., 41:44]], axis=-1)


def _unslice_gr1_44(x: np.ndarray) -> np.ndarray:
    """Scatter a 29-dim model action back to the raw 44-dim layout.

    Non-active joints (legs, neck) are filled with zeros. For real-robot
    deployment you likely want to hold the current joint state for the
    non-active dims instead.
    """
    x = np.asarray(x)
    out_shape = x.shape[:-1] + (44,)
    out = np.zeros(out_shape, dtype=x.dtype)
    out[..., 0:13] = x[..., 0:13]     # left_arm + left_hand
    out[..., 22:35] = x[..., 13:26]   # right_arm + right_hand
    out[..., 41:44] = x[..., 26:29]   # waist
    return out


def make_gr1_example() -> dict:
    """Creates a random input example for the GR-1 policy."""
    return {
        "observation/state": np.random.rand(44).astype(np.float32),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "pick up the bottled water, place it into the cabinet and close the cabinet",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GR1Inputs(transforms.DataTransformFn):
    """Map GR-1 LeRobot records into the pi0 input schema.

    - state: 44 -> 29 (arms + hands + waist)
    - action: (T, 44) -> (T, 29) (training only)
    - image: single ego view -> base_0_rgb, wrists zero-padded with mask False
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        # Pi0 expects three image slots; pad wrists with zeros.
        inputs = {
            "state": _slice_gr1_29(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # For pi0 / pi0.5 (flow-matching) we mask padded images; pi0-FAST
                # requires all slots unmasked.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _slice_gr1_29(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GR1Outputs(transforms.DataTransformFn):
    """Strip model-side action padding and return the 29-dim GR-1 action chunk.

    Leaves reconstruction back to the full 44-dim robot layout to the caller
    (e.g. the deployment wrapper), since that typically needs current-state info.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :GR1_ACTIVE_DIM])}
