from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LETTERBOX_FILL = (114, 114, 114)

_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_RESAMPLE = {
    InterpolationMode.NEAREST: _RESAMPLING.NEAREST,
    InterpolationMode.BILINEAR: _RESAMPLING.BILINEAR,
    InterpolationMode.BICUBIC: _RESAMPLING.BICUBIC,
    InterpolationMode.LANCZOS: _RESAMPLING.LANCZOS,
}


@dataclass(frozen=True)
class SquarePadResize:
    target_size: int
    fill: tuple[int, int, int] = LETTERBOX_FILL
    interpolation: InterpolationMode = InterpolationMode.BICUBIC
    scale_range: tuple[float, float] | None = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("SquarePadResize expects a PIL image input")

        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")

        fit_scale = self.target_size / max(width, height)
        if self.scale_range is not None:
            low, high = self.scale_range
            jitter = random.uniform(low, high)
            fit_scale *= min(1.0, max(0.05, jitter))

        new_width = max(1, int(round(width * fit_scale)))
        new_height = max(1, int(round(height * fit_scale)))
        resample = _PIL_RESAMPLE.get(self.interpolation, _RESAMPLING.BICUBIC)

        resized = image.resize((new_width, new_height), resample=resample)
        canvas = Image.new("RGB", (self.target_size, self.target_size), color=self.fill)

        pad_left = (self.target_size - new_width) // 2
        pad_top = (self.target_size - new_height) // 2
        canvas.paste(resized, (pad_left, pad_top))
        return canvas


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            SquarePadResize(image_size, fill=LETTERBOX_FILL, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
