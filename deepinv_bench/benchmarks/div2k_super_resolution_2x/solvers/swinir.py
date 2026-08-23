from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = "SwinIR_LightweightSR"

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        self.model = dinv.models.SwinIR(
            img_size=64,
            in_chans=3,
            embed_dim=60,
            depths=(6, 6, 6, 6),
            num_heads=(6, 6, 6, 6),
            window_size=8,
            mlp_ratio=2,
            upscale=2,
            img_range=1.0,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        )
        weights_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
        pretrained_weights = dinv.models.utils.load_state_dict_from_url(
            weights_url, map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(pretrained_weights["params"])
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
