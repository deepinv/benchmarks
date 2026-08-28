from benchopt import BaseSolver

import torch
import deepinv as dinv

WEIGHTS_BASE_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"

# Architecture and pretrained weights of official SwinIR x2 variants.
VARIANTS = {
    "lightweight": dict(
        kwargs=dict(
            embed_dim=60,
            depths=(6, 6, 6, 6),
            num_heads=(6, 6, 6, 6),
            upsampler="pixelshuffledirect",
        ),
        weights="002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
    ),
    "medium": dict(
        kwargs=dict(
            embed_dim=180,
            depths=(6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6),
            upsampler="pixelshuffle",
        ),
        weights="001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    ),
}


class Solver(BaseSolver):
    name = "SwinIR"

    parameters = {
        "variant": ["lightweight", "medium"],
    }

    def set_objective(self, train_dataset=None, physics=None):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        variant = VARIANTS[self.variant]
        self.model = dinv.models.SwinIR(
            img_size=64,
            in_chans=3,
            window_size=8,
            mlp_ratio=2,
            upscale=2,
            img_range=1.0,
            resi_connection="1conv",
            pretrained=None,
            **variant["kwargs"],
        )
        pretrained_weights = dinv.models.utils.load_state_dict_from_url(
            WEIGHTS_BASE_URL + variant["weights"],
            map_location=lambda storage, loc: storage,
        )
        self.model.load_state_dict(pretrained_weights["params"])
        self.model = self.model.to(device)
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
