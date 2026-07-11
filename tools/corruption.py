import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_corruption(image, corruption=None, severity=0):
    if corruption is None or severity == 0:
        return image

    if severity not in [1, 2, 3]:
        raise ValueError("severity must be one of 0, 1, 2, 3")

    if corruption == "gaussian_noise":
        return gaussian_noise(image, severity)
    if corruption == "motion_blur":
        return motion_blur(image, severity)
    if corruption == "brightness":
        return brightness(image, severity)
    if corruption == "rotation":
        return rotation(image, severity)
    if corruption == "translation":
        return translation(image, severity)
    if corruption == "contrast":
        return contrast(image, severity)
    if corruption == "jpeg_compression":
        return jpeg_compression(image, severity)
    if corruption == "downsample_upsample":
        return downsample_upsample(image, severity)

    raise ValueError(f"Unknown corruption: {corruption}")


def gaussian_noise(image, severity):
    stds = [8, 16, 32]
    std = stds[severity - 1]

    x = np.array(image).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=std, size=x.shape)
    x = x + noise
    x = np.clip(x, 0, 255).astype(np.uint8)

    return Image.fromarray(x)


def motion_blur(image, severity):
    radii = [1, 2, 4]
    radius = radii[severity - 1]

    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness(image, severity):
    factors = [0.85, 0.65, 0.45]
    factor = factors[severity - 1]

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def rotation(image, severity):
    """Rotate with reflected padding, then crop back to the original size."""
    angles = [5, 15, 30]
    angle = angles[severity - 1]
    x = np.asarray(image)
    height, width = x.shape[:2]
    pad = max(height, width) // 2
    pad_width = ((pad, pad), (pad, pad))
    if x.ndim == 3:
        pad_width += ((0, 0),)
    padded = Image.fromarray(np.pad(x, pad_width, mode="reflect"))
    rotated = padded.rotate(angle, resample=Image.BILINEAR, expand=False)
    left = (rotated.width - width) // 2
    top = (rotated.height - height) // 2
    return rotated.crop((left, top, left + width, top + height))


def translation(image, severity):
    """Translate the full image down/right, filling exposed pixels by reflection."""
    ratios = [0.05, 0.10, 0.20]
    ratio = ratios[severity - 1]
    x = np.asarray(image)
    height, width = x.shape[:2]
    dx = max(1, round(width * ratio))
    dy = max(1, round(height * ratio))
    pad_width = ((dy, 0), (dx, 0))
    if x.ndim == 3:
        pad_width += ((0, 0),)
    shifted = np.pad(x, pad_width, mode="reflect")[:height, :width]
    return Image.fromarray(shifted)


def contrast(image, severity):
    factors = [0.75, 0.5, 0.3]
    factor = factors[severity - 1]

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def jpeg_compression(image, severity):
    qualities = [60, 35, 15]
    quality = qualities[severity - 1]

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def downsample_upsample(image, severity):
    scales = [0.75, 0.5, 0.25]
    scale = scales[severity - 1]

    width, height = image.size
    small_width = max(1, int(width * scale))
    small_height = max(1, int(height * scale))

    small = image.resize((small_width, small_height), Image.BILINEAR)
    restored = small.resize((width, height), Image.BILINEAR)

    return restored
