import torch
import numpy as np
import lpips as lp
from torch import nn

try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity as compare_ssim

# Full-image evaluation follows Instant-NVR.


class Evaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = lp.LPIPS(net="vgg", verbose=False).cuda()
        self.loss_fn.eval()
        for p in self.loss_fn.parameters():
            p.requires_grad_(False)

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def forward(self, rgb, rgb_gt):
        # rgb: B,H,W,3
        assert len(rgb) == 1 and len(rgb_gt) == 1

        img_pred = rgb[0].detach().cpu().numpy()
        img_gt = rgb_gt[0].detach().cpu().numpy()

        psnr = self.psnr_metric(img_pred.reshape(-1, 3), img_gt.reshape(-1, 3))
        lpips = (
            self.loss_fn(
                torch.tensor(
                    img_pred.transpose((2, 0, 1)), dtype=torch.float, device="cuda"
                )[None],
                torch.tensor(
                    img_gt.transpose((2, 0, 1)), dtype=torch.float, device="cuda"
                )[None],
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        ssim = compare_ssim(img_pred, img_gt, channel_axis=2)

        return {
            "psnr": torch.Tensor([float(psnr)]).squeeze(),
            "ssim": torch.Tensor([float(ssim)]).squeeze(),
            "lpips": torch.Tensor([float(lpips)]).squeeze(),
        }
