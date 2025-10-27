import torch
from torch import Tensor
from typing import Optional


class QwenAttributeModifier():
    def __init__(self):
        self.cache_position_last: Optional[int] = None  # 用于记录上一次 forward 过程生成的 position id 最后是几
    
    def reset(self):
        """用于在训练时每调用一次 forward 就执行一次 reset 操作"""
        self.cache_position_last = None
    
    def update_position_info(self, position_ids: Optional[Tensor], cache_position: Optional[Tensor], input_ids: Tensor, is_first_input: bool):
        """
        Returns:
            position_ids, cache_position
        """
        if is_first_input:  # 第一次输入
            if cache_position is not None:
                cache_position = torch.arange(input_ids.shape[-1], device=cache_position.device, dtype=cache_position.dtype, requires_grad=cache_position.requires_grad)
                self.cache_position_last = cache_position[-1].item()
            else:
                self.cache_position_last = None
            position_ids = None  # 清空 position ids, 留给 qwen 自动生成
        elif self.cache_position_last is not None:  # 随后的 autoregressive 输入
            self.cache_position_last += 1
            cache_position = torch.tensor([self.cache_position_last], device=cache_position.device, dtype=cache_position.dtype, requires_grad=cache_position.requires_grad)

        return position_ids, cache_position
