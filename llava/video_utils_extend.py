## Adopted from qwen-vl-utils ##

from PIL import Image
import math
import torch

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    assert isinstance(ele["video"], (list, tuple))
    process_info = ele.copy()
    process_info.pop("type", None)
    process_info.pop("video", None)
    images = [
        fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
        for video_element in ele["video"]
    ]
    nframes = ceil_by_factor(len(images), FRAME_FACTOR)
    if len(images) < nframes:
        images.extend([images[-1]] * (nframes - len(images)))
    if return_video_sample_fps:
        return images, process_info.pop("fps", 2.0)
    return images


## Video Clipping ##

from typing import Union

def _clip_video_randomly(videos: list[torch.Tensor], fixed_count: int = 4):
    """
    Returns:
        (clipped_videos, video_clipping_factors) (tuple[list[Tensor], list[int]]):
            for each original video, their clipped result always satisfies `len(clipped_videos) == fixed_count`
    """
    assert isinstance(videos, list), f"For now, only a list of tensors is supported as input videos' format"
    if len(videos) <= 0:
        return videos
    assert all(isinstance(vid, torch.Tensor) for vid in videos), f"For now, only a list of tensors is supported as input videos' format"

    import random
    
    all_clips = []
    clipping_factors = []
    
    for vid_idx, video in enumerate(videos):
        # 跳过空视频
        if video.numel() == 0:
            continue
        
        num_frames, channels, height, width = video.shape
        
        # 小于2帧或者切分数<=1则直接返回 (无相邻帧的相似度)
        if num_frames <= 1 or fixed_count <= 1:
            all_clips.append(video)
            clipping_factors.append(1)
            continue

        split_indices = random.sample(range(1, num_frames), fixed_count - 1)
        all_splits = sorted(set(split_indices))
        
        # 没有切分点则直接添加整个视频
        if not all_splits:
            all_clips.append(video)
            clipping_factors.append(1)
            continue
        
        # 开始切分
        clips_cnt = 0
        start_idx = 0
        for split_idx in all_splits:
            # 确保每个片段至少包含1帧
            if split_idx > start_idx:
                # 确保分割出来的片段有偶数帧 (可能切分出来的片段之间有重复帧，但可以不用管)
                even_start_idx, even_split_idx = start_idx, split_idx
                if (split_idx - start_idx) % 2 != 0:
                    if start_idx > 0:
                        even_start_idx -= 1
                    else:
                        even_split_idx += 1
                
                clip = video[even_start_idx:even_split_idx]
                all_clips.append(clip)
                clips_cnt += 1
                start_idx = split_idx
        
        # 添加最后的片段
        if start_idx < num_frames:
            if (num_frames - start_idx) % 2 != 0:  # 确保偶数帧
                start_idx -= 1
            clip = video[start_idx:]
            all_clips.append(clip)
            clips_cnt += 1
        
        clipping_factors.append(clips_cnt)
    
    return all_clips, clipping_factors


def clip_video(video: Union[torch.Tensor, list[Image.Image]], fixed_count: int = 4):
    """
    Returns:
        (clipped_video, clipping_factor) (list[Tensor], int):
            `clipped_video` is all the clip tensors after the original video is clipped, each tensor is shaped like `[T, C, H, W]`;
            `clipping_factor` is how many clips are produced after clipping
    """
    from PIL import Image
    import numpy as np
    
    is_pil = isinstance(video, list) and all(isinstance(img, Image.Image) for img in video)

    if is_pil:
        pil_images = video
        tensor_frames = []
        for img in pil_images:
            # 转换为numpy数组
            np_img = np.array(img.convert("RGB"))
            # 转换为Tensor并归一化
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float()  # [C, H, W]
            tensor_frames.append(tensor_img)
        video_tensor = torch.stack(tensor_frames)
    else:
        video_tensor = video.clone().detach()
    
    clipped_video, clipping_factor = _clip_video_randomly([video_tensor], fixed_count)
    assert len(clipping_factor) == 1
    clipping_factor = clipping_factor[0]

    return clipped_video, clipping_factor