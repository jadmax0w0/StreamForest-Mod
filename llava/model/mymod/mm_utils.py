import os
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import Optional, Union, List, Tuple, Literal, Callable, Type


## Debug printing ##

def check_rank(rank: Union[int, List[int]]):
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        if isinstance(rank, int):
            rank = [rank]
        return (local_rank in rank)
    except KeyError:
        return True  # 单卡训练


def printr(rank: Union[int, List[int]], *msg, as_str: bool = False, to_file: Optional[str] = None, **kwargs):
    def goprint(txt):
        if isinstance(to_file, str):
            try:
                with open(to_file, mode="a") as f:
                    f.write(txt + "\n")
            except Exception as e:
                print(f"Warning: printr wanted to write to file {to_file}, but an error occured: {e}")
        
        if as_str:
            return txt
        print(txt, **kwargs)
    
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        if isinstance(rank, int):
            rank = [rank]
        if local_rank in rank:
            goprint(f"[rank {local_rank}] " + " ".join(str(m) for m in msg))
    except KeyError:
        goprint(" ".join(str(m) for m in msg))
    return None


## Hooks ##

def forward_anomaly_hook(module_name):
    def hook(module: nn.Module, input, output):
        name = module.__class__.__name__

        if not isinstance(output, (Tuple, List)):
            output = [output]
        
        for t in output:
            if isinstance(t, Tensor):
                if torch.isinf(t).any() or torch.isnan(t).any():
                    printr([0], f"anomaly found at module {name}{('/' + module_name) if module_name != '' else ''}\'s output")
    return hook


def backward_grad_hook(param_name):
    def hook(grad: Tensor):
        printr(0, f"{param_name} grad mean: {grad.mean().clone().detach().item()}, std: {grad.std().clone().detach().item()}, max: {grad.abs().max().clone().detach().item()}")
    return hook


## Maths ##

def clamp(val, mn, mx):
    return min(max(val, mn), mx)


## Output scaling (for residual connection) ##

class OutputScaling(nn.Module):
    def __init__(self, init_value: float = 1e-3, init_bias: Optional[float] = None, dim: int = 1, **kwargs):
        super().__init__(**kwargs)
        init_value = float(init_value)
        init_bias = float(init_bias) if init_bias is not None else None
        self.init_value = init_value
        self.init_bias = init_bias
        self.a = nn.Parameter(torch.full((dim, ), init_value))
        if init_bias is not None:
            self.b = nn.Parameter(torch.full((dim, ), init_bias))
    
    def weight_init(self):
        nn.init.constant_(self.a.data, self.init_value)
        if self.init_bias is not None:
            nn.init.constant_(self.b.data, self.init_bias)
    
    def forward(self, x: Tensor, force_zero: bool = False):
        # Scale NaN or Inf tensors directly to 0; or scale to 0 if specified
        if force_zero or torch.isinf(x).any() or torch.isnan(x).any():
            return torch.zeros_like(x, requires_grad=x.requires_grad)
        return (self.a * x + self.b) if self.init_bias is not None else (self.a * x)


## MLP ##

class MLPProj(nn.ModuleList):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, non_linearity: Type = nn.GELU):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.use_non_linearity = non_linearity is not None
        self.non_linearity = non_linearity

        self.initialized = False

        modules = []
        for i in range(num_layers):
            modules.extend(self._build_linear_layer_suite(i))

        super().__init__(modules)
    
    def _build_linear_layer_suite(self, layer_id: int):
        if layer_id == 0:
            return nn.Linear(self.in_dim, self.hidden_dim), self.non_linearity() if self.use_non_linearity else nn.Identity()
        elif layer_id == self.num_layers - 1:
            return nn.Linear(self.hidden_dim, self.out_dim),
        else:
            return nn.Linear(self.hidden_dim, self.hidden_dim), self.non_linearity() if self.use_non_linearity else nn.Identity()
    
    def weight_init(self, init_func: Optional[Callable] = None, **init_args):
        if self.initialized:
            print(f"{self.__class__.__name__} already initialized, skipping")
            return
        
        if init_func is None:  # when calling automatically by generate_model_weight, init_func will be none
            print(f"no initialization function is given while calling {self.__class__.__name__}'s weight_init method, using kaiming as normal")
            init_func = nn.init.kaiming_normal_
            init_args = dict(mode='fan_out', nonlinearity='relu')
        
        for module in self:
            if isinstance(module, nn.Linear):
                init_func(module.weight, **init_args)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.initialized = True
    
    def forward(self, x: Tensor):
        h = x
        for module in self:
            h = module(h)
        return h


## Extraction from original inputs_embeds ##

def extract_inputs_segments(
        inputs_embeds: Tensor,
        prompt_mask: Tensor,
        retain_videos_locations: bool = True,
        video_grid_thw: Optional[Tensor] = None,
        spatial_merge_size: Optional[int] = None,
):
    """
    Args:
        inputs_embeds (Tensor): `[1, seq_len, model_dim]`
        prompt_mask (Tensor): `[1, seq_len, model_dim]` or `[1, seq_len]`
        retain_videos_locations (bool): 为 true，则在最终输出里面对应原序列中视频的位置放置占位符，占位符用于标记每处视频位置各有多少段视频
        video_grid_thw (Optional[Tensor]): `[video_count, 3]` (frame_count, height, width)
    Returns:
        list[Tensor]-or-list[Tensor|int]: 包含文本段张量 (和视频占位符整数 if `retain_videos_locations == True`)。For tensors, each is shaped like `[sub_seq_len, model_dim]`.
    """
    # 移除 batch 维度
    inputs_embeds = inputs_embeds.squeeze(0)  # [seq_len, model_dim]
    if prompt_mask.dim() == 2:
        prompt_mask = prompt_mask.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
    prompt_mask = prompt_mask.squeeze(0)  # [seq_len, model_dim]
    seq_len = inputs_embeds.shape[0]
    
    # 获取所有 True 的位置索引
    prompt_mask = prompt_mask.any(dim=1).to(device='cpu')  # [seq_len]
    indices = torch.where(prompt_mask)[0]
    
    # 如果没有 True 值，返回空列表
    if indices.numel() == 0:
        return []
    
    # 处理单个 True 的情况
    if indices.numel() == 1:
        return [inputs_embeds[indices]]
    
    # 计算索引间的差值
    diffs = indices[1:] - indices[:-1]
    
    # 找到连续段的分界点（差值大于1表示不连续）
    break_points = torch.where(diffs > 1)[0] + 1
    # print(f"{break_points=}")
    
    # 分组连续的索引
    segment_index_groups = torch.tensor_split(indices, break_points.tolist())
    
    # 提取每组索引对应的嵌入向量
    segments = [inputs_embeds[group] for group in segment_index_groups]

    # 如果不需要保留视频位置占位符，直接返回文本段列表
    if not retain_videos_locations:
        return segments
    
    # 检查必须提供视频网格信息
    if video_grid_thw is None or spatial_merge_size is None:
        raise ValueError(f"{video_grid_thw.__qualname__} and {spatial_merge_size.__qualname__} must be provided when {retain_videos_locations.__qualname__} is True")
    
    # 计算每个视频的token数量：frame_count * height/mergesize * width/mergesize
    # video_grid_thw = torch.cat([thw // torch.tensor([1, spatial_merge_size, spatial_merge_size]) for thw in video_grid_thw])
    video_token_counts = torch.prod(video_grid_thw, dim=1)
    video_token_counts = video_token_counts // (spatial_merge_size ** 2)
    total_video_tokens = torch.sum(video_token_counts).item()
    
    # 获取文本段边界信息 (start, end)
    segment_boundaries = []
    for group in segment_index_groups:
        start_idx = group[0].item()
        end_idx = group[-1].item()
        segment_boundaries.append((start_idx, end_idx))
    # print(f"{segment_boundaries=}")
    
    # 计算文本段之间的间隙（gap）区间
    gap_intervals = []
    # 第一个文本段之前的间隙 (如果有)
    first_start = segment_boundaries[0][0]
    if first_start > 0:
        gap_intervals.append((0, first_start - 1))
    
    # 文本段之间的间隙
    for i in range(len(segment_boundaries) - 1):
        prev_end = segment_boundaries[i][1]
        next_start = segment_boundaries[i+1][0]
        gap_intervals.append((prev_end + 1, next_start - 1))
    
    # 最后一个文本段之后的间隙 (如果有)
    last_end = segment_boundaries[-1][1]
    if last_end < seq_len - 1:
        gap_intervals.append((last_end + 1, seq_len - 1))
    
    # print(f"{gap_intervals=}")
    
    # 分配视频到各个间隙区间
    video_segment_counts = [0] * len(gap_intervals)
    video_idx = 0
    
    for gap_idx, (start, end) in enumerate(gap_intervals):
        gap_length = end - start + 1
        gap_tokens_used = 0
        
        # 尝试向间隙中逐个添加视频
        while video_idx < len(video_token_counts):
            video_token_count = video_token_counts[video_idx].item()
            
            # 检查添加后是否会超出间隙容量
            if gap_tokens_used + video_token_count <= gap_length:
                video_segment_counts[gap_idx] += 1
                gap_tokens_used += video_token_count
                video_idx += 1
            else:
                # 如果当前视频无法放入，跳到下一个间隙
                break
    # print(f"{video_segment_counts=}")
    
    # 构建最终结果列表（文本段 + 视频占位符）
    result = []
    idx = 0
    prompt_first = segment_boundaries[0][0] == 0
    while idx < len(segments) and idx < len(video_segment_counts):
        if prompt_first:
            result.append(segments[idx])
            result.append(video_segment_counts[idx])
        else:
            result.append(video_segment_counts[idx])
            result.append(segments[idx])
        idx += 1
    while idx < len(segments):
        result.append(segments[idx])
        idx += 1
    while idx < len(video_segment_counts):
        result.append(video_segment_counts[idx])
        idx += 1
    
    return result


def extract_videos_embeds(
        inputs_embeds: Optional[torch.FloatTensor],
        media_mask: Optional[torch.BoolTensor],
        grid_thw: Optional[torch.LongTensor] = None,  # [media_count, frame_count, pixel_height, pixel_width]
        spatial_merge_size: int = 1,
        keep_raw_video_embeds: bool = False,
):
    """
    从 inputs_embeds 中提取出视觉 tokens，并将其转化为更符合直觉的形状

    Args:
        inputs_embeds (Optional[torch.FloatTensor]): `[batch_size, seq_len, embed_dim]`
        media_mask (Optional[torch.BoolTensor]): `[batch_size, seq_len, embed_dim]` or `[batch_size, seq_len]`
        grid_thw (Optional[torch.LongTensor]): `[media_count, frame_count, height, width]`
        spatial_merge_size (int): 在外层大模型处理视觉内容的过程中，将像素按组合并时，沿着长 & 宽方向每多少个像素合并为一组
        keep_raw_video_embeds (bool): 为 True 则同时返回从 inputs_embeds 中提取出来的原始视觉嵌入 `[batch_size, vid_token_len, embed_dim]`；否则只返回 `list[Tensor]`，每个元素为 `[frame_count, height, width, embed_dim]`

    Returns:
        list[torch.Tensor]: `[frame_count, height, width, embed_dim] x video_count` with or without `[batch_size, vid_token_len, embed_dim]`
    """
    if media_mask is None or not media_mask.any() or inputs_embeds is None:
        raise ValueError("No valid media_mask/inputs_embeds found")

    batch_size = inputs_embeds.shape[0]
    embed_dim = inputs_embeds.shape[2]
    assert batch_size == 1, "Only batchsize == 1 is supported for now"  # TODO: for other batchsizes

    # Clear batchsize dim
    inputs_embeds = inputs_embeds.squeeze(0)  # -> [seq_len, embed_dim]
    media_mask = media_mask.squeeze(0)  # -> [seq_len, embed_dim]
    if media_mask.dim() == 1:
        media_mask = media_mask.unsqueeze(-1).expand(-1, embed_dim)

    media_count = len(grid_thw)
    media_sizes = [(int(thw[0]), int(thw[1] // spatial_merge_size), int(thw[2] // spatial_merge_size)) for thw in grid_thw]  # [(frame_cnt, patch_vertical_cnt, patch_horizontal_cnt)]
    media_patch_counts = [t * v * h for t, v, h in media_sizes]
    # print(f"{media_count=}, {media_sizes=}, {media_patch_counts=}")

    media_mask = media_mask.any(dim=1).to(device='cpu')  # removing the `embed_dim` in `media_mask` => [seq_len]

    # Selecting only media tokens
    inputs_embeds_media = inputs_embeds[media_mask]  # [media_len, embed_dim]
    
    # Reshaping `inputs_embeds_media`
    reshaped_inputs_embeds_media = []
    for i in range(media_count):
        media_embeds_start, media_embeds_end = sum(media_patch_counts[:i]), sum(media_patch_counts[:i + 1])
        # print(f"{media_embeds_start=}, {media_embeds_end=}")
        media_embeds = inputs_embeds_media[media_embeds_start:media_embeds_end]
        media_embeds = media_embeds.view(*media_sizes[i], -1)
        reshaped_inputs_embeds_media.append(media_embeds)
    
    if keep_raw_video_embeds:
        return reshaped_inputs_embeds_media, inputs_embeds_media.unsqueeze(0)
    else:
        return reshaped_inputs_embeds_media


## Divide images of different resolution into windows of unified count ##

def _distribute_windows_side_len(full_len: int, windows_cnt: int):
    """
    Returns:
        list[int]: `windows_cnt` 个 values of windows sizes, all summing up to `full_len`
    """
    if windows_cnt <= 0:
        raise ValueError(f"{windows_cnt=} is not supported")
    
    # 计算基值和余数
    base = full_len // windows_cnt
    remainder = full_len % windows_cnt
    
    # 创建序列：大多数值为基值base，部分为base+1（由余数决定）
    result = []
    for i in range(windows_cnt):
        # 前remainder个数增加1，确保均匀分布
        if i < remainder:
            result.append(base + 1)
        else:
            result.append(base)
    
    assert sum(result) == full_len
    
    result.sort()
    return result


def _get_window_index(clip_embeds: torch.Tensor, num_windows_h: int, num_windows_w: int):
    """
    Returns:
        (permute_indices, windows_cumulative_seqlen): `tuple[Tensor, Tensor]`
    """
    grid_t, grid_h, grid_w, embed_dim = tuple(clip_embeds.shape)
    
    # 每行/列的窗口尺寸
    window_sizes_h = _distribute_windows_side_len(grid_h, num_windows_h)
    window_sizes_w = _distribute_windows_side_len(grid_w, num_windows_w)

    # 索引 [t, h, w]
    base_index = torch.arange(grid_t * grid_h * grid_w, device='cpu').reshape(grid_t, grid_h, grid_w)
    
    window_indices = []   # 收集各窗口的索引
    window_seqlens = []   # 记录各窗口的元素数量
    total_index_offset = 0  # 多帧时的索引偏移量

    # 遍历所有帧和时间步
    for t in range(grid_t):
        # 遍历行窗口
        row_start = 0
        for win_h in window_sizes_h:
            # 遍历列窗口
            col_start = 0
            for win_w in window_sizes_w:
                # 截取当前窗口区域
                window = base_index[t, row_start:row_start+win_h, col_start:col_start+win_w]
                
                # 展平并添加到索引列表
                flat_indices = window.contiguous().view(-1)
                window_indices.append(flat_indices)
                
                # 记录当前窗口元素数量
                win_seqlen = win_h * win_w
                window_seqlens.append(win_seqlen)
                
                col_start += win_w  # 移动到下一列窗口
            row_start += win_h      # 移动到下一行窗口
        total_index_offset += grid_h * grid_w  # 增加下一帧的索引偏移

    # 合并所有窗口索引 [总元素数]
    combined_indices = torch.cat(window_indices, dim=0)

    window_seqlens = [0] + window_seqlens
    window_seqlens = torch.tensor(window_seqlens, dtype=base_index.dtype, device='cpu')
    
    return combined_indices, window_seqlens.cumsum(0)


def combine_frames_patches_into_embed_obsolete(clip_embeds: torch.Tensor, num_windows_per_side: int, flatten_frm_feats: bool = False):
    """
    将不同分辨率的帧以窗口池化的方式转换为固定大小的 embedding.
    Args:
        clip_embeds: shape `[frame_count, patch_count_h, patch_count_w, model_dim]`
    Returns:
    >>> if flatten_frm_feats:
    >>>     [frame_cnt, patch_cnt_v, patch_cnt_h, model_dim] -> [frame_cnt, window_cnt * model_dim]
    >>> else:
    >>>     [frame_cnt, patch_cnt_v, patch_cnt_h, model_dim] -> [frame_cnt, num_wins_side, num_wins_side, model_dim]
    """
    frame_count, patch_cnt_h, patch_cnt_w, model_dim = clip_embeds.shape

    # 处理帧本身的分辨率小于分窗口后分辨率的情况 (直接插值)
    if patch_cnt_h < num_windows_per_side or patch_cnt_w < num_windows_per_side:
        clip_embeds = F.interpolate(
            clip_embeds.permute(0, 3, 1, 2),
            size=(max(patch_cnt_h, num_windows_per_side), max(patch_cnt_w, num_windows_per_side))
        )
        clip_embeds = clip_embeds.permute(0, 2, 3, 1)
    
    window_index, cu_window_seqlens = _get_window_index(clip_embeds, num_windows_per_side, num_windows_per_side)

    clip_embeds = clip_embeds.flatten(0, -2)  # [frame_cnt * patch_cnt_v * patch_cnt_h, model_dim]
    clip_embeds = clip_embeds[window_index]
    merged_video_embeds = [torch.mean(clip_embeds[s:t], dim=0) for s, t in zip(cu_window_seqlens[:-1], cu_window_seqlens[1:])]  # [model_dim] x (frame_cnt * window_cnt)
    merged_video_embeds = torch.stack(merged_video_embeds, dim=0)  # [frame_cnt * window_cnt, model_dim]
    if not flatten_frm_feats:
        return merged_video_embeds.view(frame_count, num_windows_per_side, num_windows_per_side, model_dim)  # [frm_cnt, win_side_cnt, win_side_cnt, model_d]
    merged_video_embeds = merged_video_embeds.view(frame_count, -1)  # [frame_cnt, window_cnt * model_dim]
    return merged_video_embeds


def downsample_frames(clip_embeds: torch.Tensor, size: Union[int, Tuple[int, int]]):
    """
    Args:
        clip_embeds: shape `[frame_count, patch_count_h, patch_count_w, model_dim]`
    Returns:
        clip_embeds_new: `[frame_cnt, size[0], size[1], model_dim]`
    """
    if isinstance(size, int):
        size = (size, size)
    clip_embeds_new = F.interpolate(
        clip_embeds.permute(0, 3, 1, 2),                    # [frame_count, model_dim, patch_h, patch_w]
        size=size,
    )
    clip_embeds_new = clip_embeds_new.permute(0, 2, 3, 1)   # [frame_count, #win, #win, model_dim]
    return clip_embeds_new


## Formatting new inputs ##

def _cat_mask_values(mask: Tensor, value: int, length: int, has_embed_dim: bool = True, batchsize: Optional[int] = None, embed_dim: Optional[int] = None):
    """
    Args:
        mask (Tensor): shaped `[B, N, D]` or `[B, N]`
    Returns:
        cated_mask (Tensor): shaped `[B, N+length, D]` or `[B, N+length]`
    """
    if length <= 0:
        return mask
    if has_embed_dim:
        assert mask.dim() == 3 or mask.numel() <= 0
        try:
            B, N, D = mask.shape
            batchsize, embed_dim = B, D
        except ValueError:
            pass
        return torch.cat([mask, torch.full((batchsize, length, embed_dim), value, device=mask.device, dtype=mask.dtype, requires_grad=mask.requires_grad)], dim=1)
    else:
        assert mask.dim() == 2 or mask.numel() <= 0
        try:
            B, N = mask.shape
            batchsize, embed_dim = B, None
        except ValueError:
            pass
        return torch.cat([mask, torch.full((batchsize, length), value, device=mask.device, dtype=mask.dtype, requires_grad=mask.requires_grad)], dim=1)


def format_new_inputs(
        inputs_segments: List[Tensor],
        labels_segments: Optional[List[Tensor]],
        mm_embeds: Tuple[Tensor],
        mm_scatter_lengths: Tuple[List[int]],
        inputs_embeds_old: Tensor,
        labels_old: Optional[Tensor],
        position_ids_old: Optional[Tensor] = None,
        attention_mask_old: Optional[Tensor] = None,
        ignore_label: Optional[int] = -100,
        mm_scatter_align: Literal["left", "right"] = "left",
        mm_scatter_from_begin: bool = False,
        mm_scatter_to_end: bool = False,
        mm_sep_tokens: Optional[List[int]] = None,
        mm_sep_embeds: Optional[Tensor] = None,
        input_ids_segments: Optional[List[Tensor]] = None,
        input_ids_old: Optional[Tensor] = None,
        video_pad_token: Optional[int] = None,
):
    """
    Args:
        inputs_segments: shape `[N, D]` x segment count
        labels_segments: shape `[N]` x segment count
        mm_embeds (tuple): shape `[B, N, D]` x multimedia content num
        mm_scatter_lengths (tuple): 每个 multimedia content 将在每两个 inputs segments 之间各分散多少个 token。如 `([1, 2], [3, 4])` 表明将第一个 mm content 划分为 1, 2 个 token 依次放到第 1/2 和第 2/3 个 inputs segments 之间、将第二个 mm content 划分为 3, 4 个 token 依次放到第 1/2 和第 2/3 个 inputs segments 之间
        mm_scatter_align: 若某个 mm content 的 scatter length list 不足以将其分散到所有 inputs segments 之间，则将其向哪边对齐。如 `mm_scatter_lengths=([1, 2], [7])`，将第二个 mm content 向左对齐，则得到 `mm_scatter_lengths=([1, 2], [7, 0])`，向右对齐则得到 `mm_scatter_lengths=([1, 2], [0, 7])`
        mm_scatter_from_begin: 是否在第一个 input segment 之前也 scatter
        mm_scatter_to_end: 是否在最后一个 input segment 之后也 scatter
        mm_sep_tokens: 每两个 mm content 之间用什么 token 隔开，比如 `[<|video_end|>, <|video_start|>]`
        mm_sep_embeds: 对应 `mm_sep_tokens` 中每个 token id 的 embedding, shape should be like `[num_sep_tokens, model_dim]`
        input_ids_segments: shape `[N]` x segment count
    Returns:
        `inputs_embeds_new`, `(labels_new if labels_old is not None else None)`, `position_ids_new`, `attention_mask_new`, `(input_ids_new if input_ids_segments is not None else None)`
    """
    mm_sep_count = 0

    def cat_mm_scatter_mask(scatter_id, batchsize, model_dim, inputs_mask, posidx_mask, labels_new, inids_new):
        nonlocal mm_sep_count
        for mmid, mm_emb in enumerate(mm_embeds):
            mask_len = mm_scatter_lengths[mmid][scatter_id]
            inputs_mask = _cat_mask_values(inputs_mask, mmid, mask_len, batchsize=batchsize, embed_dim=model_dim)
            posidx_mask.extend([mmid] * mask_len)
            if labels_old is not None:
                labels_new = _cat_mask_values(labels_new, ignore_label, mask_len, has_embed_dim=False, batchsize=batchsize)
            if inids_new is not None:
                inids_new = _cat_mask_values(inids_new, video_pad_token, mask_len, has_embed_dim=False, batchsize=batchsize, embed_dim=model_dim)
            
            # 每两个 mm content 之间添加分隔符 token
            if isinstance(mm_sep_tokens, list) and mmid < len(mm_embeds) - 1:
                sep_token_cnt = len(mm_sep_tokens)
                inputs_mask = _cat_mask_values(inputs_mask, -2, sep_token_cnt, has_embed_dim=True, batchsize=batchsize, embed_dim=model_dim)
                posidx_mask.extend([-1] * sep_token_cnt)
                if labels_old is not None:
                    labels_new = _cat_mask_values(labels_new, ignore_label, sep_token_cnt, has_embed_dim=False, batchsize=batchsize)
                if inids_new is not None:
                    for sep in mm_sep_tokens:
                        inids_new = _cat_mask_values(inids_new, sep, 1, has_embed_dim=False, batchsize=batchsize)
                mm_sep_count += 1  # 多添加了一组 mm separator token
        return inputs_mask, posidx_mask, labels_new, inids_new
    
    batchsize, seq_len_old, model_dim = inputs_embeds_old.shape

    inputs_device, inputs_dtype = inputs_embeds_old.device, inputs_embeds_old.dtype
    labels_device, labels_dtype = (labels_old.device, labels_old.dtype) if labels_old is not None else (None, None)
    posidx_device, posidx_dtype = (position_ids_old.device, position_ids_old.dtype) if position_ids_old is not None else (None, None)
    atnmsk_device, atnmsk_dtype = (attention_mask_old.device, attention_mask_old.dtype) if attention_mask_old is not None else (None, None)
    inpids_device, inpids_dtype = (input_ids_old.device, input_ids_old.dtype) if input_ids_old is not None else (None, None)

    ## 1️⃣ 首先将所有的 scatter lengths 填充至相同的长度，并保证每个 scatter lengths 加和都等于对应 mm embeds 的长度
    max_mm_scatter_count = len(inputs_segments) - 1
    if mm_scatter_from_begin:
        max_mm_scatter_count += 1
    if mm_scatter_to_end:
        max_mm_scatter_count += 1
    
    for i, curr_emb in enumerate(mm_embeds):
        # mm scatter lengths 数量少于 mm embeds 数量
        if i >= len(mm_scatter_lengths):
            i -= 1  # 这一个 mm embed 标记为“还没有见过”
            break
        curr_batchsize, curr_seq_len, curr_emb_dim = curr_emb.shape
        # 保证加和为 mm seq len
        if sum(mm_scatter_lengths[i]) < curr_seq_len:
            mm_scatter_lengths[i].append(curr_seq_len - sum(mm_scatter_lengths[i]))
        # 填充/合并至统一长度
        curr_scatter_count = len(mm_scatter_lengths[i])
        if curr_scatter_count < max_mm_scatter_count:
            if "left" in mm_scatter_align.lower():
                mm_scatter_lengths[i].extend([0] * (max_mm_scatter_count - curr_scatter_count))
            elif "right" in mm_scatter_align.lower():
                mm_scatter_lengths[i][:] = [0] * (max_mm_scatter_count - curr_scatter_count) + mm_scatter_lengths[i]
            else:
                raise ValueError(f"Unknown {mm_scatter_align=}")
        elif curr_scatter_count > max_mm_scatter_count:
            if "left" in mm_scatter_align.lower():
                mm_scatter_lengths[i][:] = mm_scatter_lengths[i][:max_mm_scatter_count - 1] + [sum(mm_scatter_lengths[i][max_mm_scatter_count - 1:])]
            elif "right" in mm_scatter_align.lower():
                mm_scatter_lengths[i][:] = [sum(mm_scatter_lengths[i][:-max_mm_scatter_count + 1])] + mm_scatter_lengths[i][-max_mm_scatter_count + 1:]
            else:
                raise ValueError(f"Unknown {mm_scatter_align=}")
    i += 1  # next unseen mm embed
    # 处理剩余的几个 mm embeds (这几个没对应的 mm scatter lengths)
    new_scatter_lengths = []
    while i < len(mm_embeds):
        _, curr_seq_len, _ = mm_embeds[i].shape
        curr_scatter_lengths = [curr_seq_len]
        if "left" in mm_scatter_align.lower():
            curr_scatter_lengths.extend([0] * (max_mm_scatter_count - 1))
        elif "right" in mm_scatter_align.lower():
            curr_scatter_lengths = [0] * (max_mm_scatter_count - 1) + curr_scatter_lengths
        else:
            raise ValueError(f"Unknown {mm_scatter_align=}")
        new_scatter_lengths.append(curr_scatter_lengths)
        i += 1
    if new_scatter_lengths:
        mm_scatter_lengths = tuple(list(mm_scatter_lengths) + new_scatter_lengths)

    ## 2️⃣ 然后开始构建 scatter mask
    inputs_mask = torch.empty(0, device=inputs_device)  # [1, new_seq_len, model_dim]
    position_ids_mask = []  # 用于辅助创建 3d rope position id (TODO: batchsize == 1)
    labels_new = None
    input_ids_new = None

    texts_embeds = torch.empty(0, device=inputs_device)
    if labels_old is not None:
        labels_new = torch.empty(0, device=labels_device, dtype=labels_dtype)
        curr_label_segid = 0
    if input_ids_segments is not None:
        input_ids_new = torch.empty(0, device=inpids_device, dtype=inpids_dtype)
        curr_inid_segid = 0
    curr_scatter_id = 0

    # 处理第一个文本段之前的 scatter
    if mm_scatter_from_begin:
        inputs_mask, position_ids_mask, labels_new, input_ids_new = cat_mm_scatter_mask(curr_scatter_id, batchsize, model_dim, inputs_mask, position_ids_mask, labels_new, input_ids_new)
        curr_scatter_id += 1
        curr_inid_segid += 1
    
    # 处理每个文本段之间需要 scatter 的 mm embed
    for segid, seg in enumerate(inputs_segments):
        # seg: 文本段，形状 [seq_len, model_dim]，无 batchsize
        # 处理文本段的 scatter mask 以及对应的 position id mask, new label
        inputs_mask = _cat_mask_values(inputs_mask, value=-1, length=seg.shape[0], batchsize=batchsize, embed_dim=model_dim)
        position_ids_mask.extend([-1] * seg.shape[0])
        if labels_old is not None:
            labels_new = torch.cat([labels_new, labels_segments[curr_label_segid].unsqueeze(0)], dim=1)
            curr_label_segid += 1
        if input_ids_segments is not None:
            input_ids_new = torch.cat([input_ids_new, input_ids_segments[curr_inid_segid].unsqueeze(0)], dim=1)
            curr_inid_segid += 1
        # 只处理每个文本段之间的 scatter
        if segid < len(inputs_segments) - 1:
            inputs_mask, position_ids_mask, labels_new, input_ids_new = cat_mm_scatter_mask(curr_scatter_id, batchsize, model_dim, inputs_mask, position_ids_mask, labels_new, input_ids_new)
            curr_scatter_id += 1
        # 顺便处理 texts embeds
        texts_embeds = torch.cat([texts_embeds, seg.unsqueeze(0)], dim=1)
    
    # 处理最后一个文本段后面的 scatter
    if mm_scatter_to_end:
        inputs_mask, position_ids_mask, labels_new, input_ids_new = cat_mm_scatter_mask(curr_scatter_id, batchsize, model_dim, inputs_mask, position_ids_mask, labels_new, input_ids_new)
        curr_scatter_id += 1
    
    ## 3️⃣ 开始做 masked scatter
    inputs_embeds_new = torch.zeros_like(inputs_mask).to(dtype=inputs_dtype, device=inputs_device)
    inputs_embeds_new = inputs_embeds_new.masked_scatter(inputs_mask == -1, texts_embeds.to(dtype=inputs_dtype, device=inputs_device))
    for mmid, mm_emb in enumerate(mm_embeds):
        inputs_embeds_new = inputs_embeds_new.masked_scatter(inputs_mask == mmid, mm_emb.to(dtype=inputs_dtype, device=inputs_device))
    if mm_sep_count > 0 and mm_sep_tokens is not None:
        assert mm_sep_embeds is not None, "You should provide both mm_sep_tokens and mm_sep_embeds"
        sep_embeds_full = [mm_sep_embeds for _ in range(mm_sep_count)]
        sep_embeds_full = torch.cat(sep_embeds_full, dim=0)
        inputs_embeds_new = inputs_embeds_new.masked_scatter(inputs_mask == -2, sep_embeds_full)

    position_ids_new = torch.arange(
        inputs_embeds_new.shape[1], device=posidx_device, dtype=posidx_dtype
    ).unsqueeze(0).unsqueeze(0).expand(3, batchsize, -1)
    attention_mask_new = torch.ones((batchsize, inputs_embeds_new.shape[1]), device=atnmsk_device, dtype=atnmsk_dtype)

    return inputs_embeds_new, (labels_new if labels_old is not None else None), position_ids_new, attention_mask_new, (input_ids_new if input_ids_segments is not None else None)


## Video clipping ##

def clip_video_with_count(videos: list[torch.Tensor], fixed_count: int = 2):
    """
    Returns:
        (clipped_videos, video_clipping_factors) (tuple[list[Tensor], list[int]]):
            for each original video, their clipped result always satisfies `len(clipped_videos) == fixed_count`
    """
    assert isinstance(videos, list), f"For now, only a list of tensors is supported as input videos' format"
    if len(videos) <= 0:
        return videos
    assert all(isinstance(vid, torch.Tensor) for vid in videos), f"For now, only a list of tensors is supported as input videos' format"
    
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
        
        # RGB 色彩归一化到[0,1]
        if video.dtype != torch.float32:
            video = video.to(torch.float32) / 255.0
        elif video.max() > 1.0:
            video = video / 255.0
        
        # 下采样加速计算
        resized_video = F.interpolate(video.permute(0, 2, 3, 1), 
                                    size=(64, 64), 
                                    mode='bilinear',
                                    align_corners=False).permute(0, 3, 1, 2)
        
        # 将帧展平为向量
        flattened_frames = resized_video.reshape(num_frames, -1)
        
        # 计算相邻帧的相似度
        current_frame = flattened_frames[:-1]
        next_frame = flattened_frames[1:]
        
        cos_similarities = F.cosine_similarity(current_frame, next_frame, dim=1)

        # 找到切分位置：找到所有相邻帧中相似度前 fixed_count-1 小的位置进行切割
        # 计算需要找出的切分点数量
        k = min(fixed_count - 1, len(cos_similarities))
        if k > 0:
            # 获取前k个最小相似度的索引
            _, least_similar_indices = torch.topk(
                cos_similarities, 
                k=k,
                largest=False
            )
            # 转换为切分点并排序去重
            split_indices = (least_similar_indices + 1).tolist()
            split_indices = sorted(set(split_indices))
        else:
            split_indices = []

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


def clip_video_randomly(videos: list[torch.Tensor], fixed_count: int = 2):
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


def update_message(target_messages: list[dict], clipping_factors: list[int]):
    from copy import deepcopy
    messages = deepcopy(target_messages)

    vid_id = 0
    for mid, msg in enumerate(messages):
        if not isinstance(msg['content'], list):
            continue
        contents_new = []
        for content in msg['content']:
            ctype = content['type'].lower()
            if "text" in ctype:
                contents_new.append(content)
            elif "video" in content['type'].lower():
                contents_new.extend([content.copy() for _ in range(clipping_factors[vid_id])])
                vid_id += 1
        msg['content'] = contents_new
    
    return messages

## Debugging modulars ##

def dummy_window_pooling(
        inputs_embeds: Tensor,
        video_mask: Tensor,
        video_grid_thw: Tensor,
        labels: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        spatial_merge_size: int = 1,
        windows_side_cnt: int = 4,
        # 生成新的 input_ids, 用于设置 rope position id
        input_ids: Optional[Tensor] = None,
        video_pad_token: Optional[int] = None,
        vision_start_token: Optional[int] = None,
        vision_end_token: Optional[int] = None,
        vision_start_embeds: Optional[Tensor] = None,
        vision_end_embeds: Optional[Tensor] = None,
):
    """
    Args:
        vision_start_embeds: shape `[1, model_dim]`
        vision_end_embeds: shape `[1, model_dim]`
    Returns:
        `inputs_embeds_new`, `(labels_new if labels_old is not None else None)`, `position_ids_new`, `attention_mask_new`, `(input_ids_new if input_ids_segments is not None else None)`, `video_grid_thw_new`
    """
    inputs_segments = extract_inputs_segments(inputs_embeds, ~video_mask, retain_videos_locations=False)
    labels_segments = extract_inputs_segments(labels, ~video_mask, retain_videos_locations=False) if labels is not None else None
    input_ids_segments = extract_inputs_segments(input_ids, ~video_mask, retain_videos_locations=False) if input_ids is not None else None
    videos_embeds = extract_videos_embeds(inputs_embeds, video_mask, video_grid_thw, spatial_merge_size)

    video_count = len(videos_embeds)
    assert video_count == video_grid_thw.shape[0], f"Extracted video count ({video_count}) should be the same as what `video_grid_thw` provided ({video_grid_thw})"

    merged_clips = []
    for vid_idx, clip_embeds in enumerate(videos_embeds):
        frame_count = clip_embeds.shape[0]

        merged_clip_embeds = downsample_frames(clip_embeds, windows_side_cnt)  # [frm_cnt, num_win_side, num_win_side, model_d]
        merged_clip_embeds = merged_clip_embeds.flatten(0, 2)  # [frm_cnt*win_cnt, model_d]

        merged_clips.append(merged_clip_embeds)
    
    merged_clips_embeds = (torch.cat(merged_clips, dim=0).unsqueeze(0), )
    clipemb_scatter_lengths = ([thw[0].item() * (windows_side_cnt ** 2) for thw in video_grid_thw], )

    use_mm_sep = vision_start_token is not None and vision_end_token is not None and vision_start_embeds is not None and vision_end_embeds is not None
    formatted_new_inputs = format_new_inputs(
        inputs_segments=inputs_segments,
        labels_segments=labels_segments,
        mm_embeds=merged_clips_embeds,
        mm_scatter_lengths=clipemb_scatter_lengths,
        inputs_embeds_old=inputs_embeds,
        labels_old=labels,
        position_ids_old=position_ids,
        attention_mask_old=attention_mask,
        input_ids_segments=input_ids_segments,
        input_ids_old=input_ids,
        video_pad_token=video_pad_token,
        mm_sep_tokens=[vision_start_token, vision_end_token] if use_mm_sep else None,
        mm_sep_embeds=torch.cat([vision_start_embeds, vision_end_embeds], dim=0) if use_mm_sep else None,
    )

    video_grid_thw_new = torch.tensor([[thw[0], windows_side_cnt * spatial_merge_size, windows_side_cnt * spatial_merge_size] for thw in video_grid_thw], dtype=video_grid_thw.dtype, device=video_grid_thw.device)

    return *formatted_new_inputs, video_grid_thw_new


## Testing ##
if __name__ == "__main__":
    # inputs_embeds = torch.arange(1, 364, dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6)
    vid1 = [[0 for _ in range(10)] for _ in range(10)]
    base_val = 10
    for row in range(10):
        for col in range(10):
            offset_val = 1
            if 0 <= row <= 1:
                offset_val += 1
            elif 2 <= row <= 3:
                offset_val += 2
            elif 4 <= row <= 6:
                offset_val += 3
            else:
                offset_val += 4
            if 0 <= col <= 1:
                offset_val += 1
            elif 2 <= col <= 3:
                offset_val += 2
            elif 4 <= col <= 6:
                offset_val += 3
            else:
                offset_val += 4
            vid1[row][col] = base_val + offset_val
    vid2 = [[0 for _ in range(5)] for _ in range(5)]
    base_val = 5
    for row in range(5):
        for col in range(5):
            offset_val = 1
            if row == 0:
                offset_val += 1
            elif row == 1:
                offset_val += 2
            elif row == 2:
                offset_val += 3
            else:
                offset_val += 4
            if col == 0:
                offset_val += 1
            elif col == 1:
                offset_val += 2
            elif col == 2:
                offset_val += 3
            else:
                offset_val += 4
            vid2[row][col] = base_val + offset_val
    vid1 = torch.tensor(vid1, dtype=torch.float).flatten().tolist()
    vid2 = torch.tensor(vid2, dtype=torch.float).flatten().tolist()

    input_ids = torch.tensor([1, 1, 4, 5, 1] + [-2] * (len(vid1) * 3) + [7, 8] + [-3] * (len(vid2) * 2) + [1, 9, 1, 9, 8, 1], dtype=torch.int).unsqueeze(0)
    inputs_embeds = torch.tensor([1, 2, 3, 4, 5] + vid1 * 3 + [6, 7] + vid2 * 2 + [1, 1, 4, 5, 1, 4], dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6)
    video_mask = torch.tensor([0] * 5 + [1] * 300 + [0] * 2 + [1] * 50 + [0] * 6, dtype=torch.bool).unsqueeze(0)
    labels = torch.tensor(([-100] * 357 + [11, 45, 14, 19, 19, 810]), dtype=torch.int).unsqueeze(0)
    video_grid_thw = torch.tensor([[3, 10, 10], [2, 5, 5]], dtype=torch.int)  # [vid_cnt, 3]

    inputs_embeds, labels, position_ids, attention_mask, input_ids, video_grid_thw = dummy_window_pooling(
        inputs_embeds, video_mask, video_grid_thw, labels,
        windows_side_cnt=4, input_ids=input_ids, video_pad_token=-5,
        vision_start_token=69, vision_end_token=96,
        vision_start_embeds=torch.tensor([[3, 3, 4, 4, 1, 8]], dtype=inputs_embeds.dtype, device=inputs_embeds.device),
        vision_end_embeds=torch.tensor([[4, 2, 7, 9, 6, 3]], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    )
    pass

    # Example usage
    input_ids = torch.arange(1, 28).unsqueeze(0)  # [1, 27]
    inputs_embeds = torch.arange(1, 28, dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6)  # [1, 27, 6]
    video_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool).unsqueeze(0)
    labels = torch.tensor(([-100] * 24 + [11, 45, 14]), dtype=torch.int).unsqueeze(0)  # [1, 27]
    video_grid_thw = torch.tensor([[1, 2, 2], [4, 1, 1], [3, 1, 3]], dtype=torch.int)  # [vid_cnt, 3]

    input_ids_segments = extract_inputs_segments(input_ids, ~video_mask, retain_videos_locations=False)
    inputs_segments_info = extract_inputs_segments(inputs_embeds, ~video_mask, retain_videos_locations=True, video_grid_thw=video_grid_thw, spatial_merge_size=1)
    inputs_segments = extract_inputs_segments(inputs_embeds, ~video_mask, retain_videos_locations=False)
    labels_segments = extract_inputs_segments(labels, ~video_mask, retain_videos_locations=False)
    reshaped_videos_embeddings, videos_embeddings = extract_videos_embeds(inputs_embeds, media_mask=video_mask, grid_thw=video_grid_thw, spatial_merge_size=1, keep_raw_video_embeds=True)

    a_embs = torch.randn(1, 10, 6) + 3.0
    b_embs = torch.randn(1, 4, 6) + 5.0
    c_embs = torch.randn(1, 20, 6) - 5.0

    inputs_embeds_new, labels_new, posidx_new, attn_new, input_ids = format_new_inputs(
        inputs_segments, labels_segments, 
        (a_embs, b_embs, c_embs), ([4], [1, 1, 1, 1]), 
        inputs_embeds, labels,
        mm_sep_tokens=[69, 96],
        mm_sep_embeds=torch.tensor([[3, 3, 4, 4, 1, 8], [4, 2, 7, 9, 6, 3]], dtype=inputs_embeds.dtype, device=inputs_embeds.device),
        input_ids_old=input_ids,
        input_ids_segments=input_ids_segments,
        video_pad_token=99,
    )
    # TODO: 现存的问题：如果第一个 mm content 的 scatter lengths 是 [4, 0]，第二个是 [1, 3]，那么在第二处 mm scatter 位置添加分隔符时，明明没有第一个 mm content 的内容，但还是会有一个分隔符
    pass