from sennet.custom_modules.models.base_model import Base3DSegmentor, SegmentorOutput
from line_profiler_pycharm import profile
import torch


class Tta3DSegmentor(Base3DSegmentor):
    def __init__(
            self,
            base_model: Base3DSegmentor,
            add_flips: bool = True,
    ):
        Base3DSegmentor.__init__(self)
        self.base_model = base_model
        self.add_flips = add_flips

    def get_name(self) -> str:
        return f"Tta({self.base_model})"

    @profile
    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        """

        :param img: torch.Tensor: (b, c, z, h, w)
        :return: SegmentorOutput:
            - pred: (b, z1, h, w)
            - take_indices: (z, ): which z channels from the input this is meant to predict, starting from 0
        """
        output_preds = []
        take_indices_start = None
        take_indices_end = None
        for i in range(img.shape[0]):
            tta_imgs = self._generate_tta_imgs(img[i])
            base_model_output = self.base_model.predict(tta_imgs)
            decoded_model_output_tensors = self._decode_model_outputs(base_model_output.pred)

            if take_indices_start is None:
                take_indices_start = base_model_output.take_indices_start
            if take_indices_end is None:
                take_indices_end = base_model_output.take_indices_end

            output_batch = torch.mean(decoded_model_output_tensors, dim=0)
            output_preds.append(output_batch)
        return SegmentorOutput(
            pred=torch.stack(output_preds, dim=0),
            take_indices_start=take_indices_start,
            take_indices_end=take_indices_end,
        )

    @profile
    def _generate_tta_imgs(self, img: torch.Tensor) -> torch.Tensor:
        """
        :param img:  (c, z, h, w)
        :return: (b, c, z, h, w): tta augmented for the model
        """
        tta_imgs = [img]
        if self.add_flips:
            h_flipped = torch.flip(img, [3])
            v_flipped = torch.flip(img, [2])
            hv_flipped = torch.flip(img, [2, 3])
            tta_imgs += [
                h_flipped,
                v_flipped,
                hv_flipped,
            ]
        return torch.stack(tta_imgs, dim=0)

    def _decode_model_outputs(self, preds: torch.Tensor) -> torch.Tensor:
        """
        :param preds:  (b, z, h, w)
        :return: (b, z, h, w): tta augmented for the model
        """
        i = 1
        if self.add_flips:
            preds[i, ...] = torch.flip(preds[i], [2])
            i += 1
            preds[i, ...] = torch.flip(preds[i], [1])
            i += 1
            preds[i, ...] = torch.flip(preds[i], [2, 1])
            i += 1
        return preds


if __name__ == "__main__":
    from sennet.custom_modules.models import SMPModel

    _device = "cuda"
    _model = SMPModel(
        version="Unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).eval().to(_device)
    _tta_model = Tta3DSegmentor(
        _model,
        add_flips=True,
    )
    _img = torch.ones((2, 1, 1, 512, 512)).to(_device)
    with torch.no_grad():
        _out = _tta_model.predict(_img)
    print(_out)
