# import sys
# sys.path.extend(['.', ".."])
import os
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Literal, Callable

from .tcn_module import MS_TCN2
from .rnn_module import GroupGRU
from .attention_module import MultiheadAttention, ThresholdMultiheadAttention, GraphLikeMultiheadAttention, MemoryAttentionDecoder, \
    IntraFrameDecoderConfig, InterFrameDecoderConfig, InterClipDecoderConfig
from .mm_utils import downsample_frames, printr, clamp, OutputScaling, MLPProj, forward_anomaly_hook, backward_grad_hook


class ShortTermMemoryItem():
    def __init__(self, t: int, h: int, w: int, data: Tensor):
        self.t = t
        self.h = h
        self.w = w
        self.data = data


class ShortTermMemory(nn.Module):
    FRAME_STORAGE_DEFAULT_TRAIN = 64
    FRAME_STORAGE_DEFAULT_TEST = 180

    def __init__(
            self,
            storage: int = 64,
            pooling_scale_side: int = 2,
            model_dim: int = 3584,
            tcn_num_PG_layers: int = 3,
            tcn_num_R_layers: int = 1,
            tcn_num_R: int = 2,
            tcn_hidden_dim: int = 512,
            rnn_num_layers: int = 3,
            rnn_hidden_dim: int = 512,
            output_channels: int = 3584,
            storage_method: Literal["clip", "frame"] = "frame",
            **kwargs
    ):
        """
        Args:
            frame_storage (int): 总共保存最后多少帧的记忆；`-1` 或 `< 0` 则保存所有片段的记忆
            pooling_scale_size (int): 沿着宽/高的方向，将每帧按多大的比例池化
            num_pooled_patch_per_side (int): 分窗口后，再卷积成几乘几的 patch embedding
        """
        super().__init__(**kwargs)
        self.clip_storage = storage
        self.frame_storage = storage  # max cumulative frame count in STM
        self.storage_method = storage_method.lower()

        self.model_dim = model_dim
        self.pooling_scale_side = pooling_scale_side
        self.frame_channels = model_dim
        self.output_channels = output_channels

        self.disabled = False

        # nn modules
        self.tcn = MS_TCN2(
            num_layers_PG=tcn_num_PG_layers,
            num_layers_R=tcn_num_R_layers,
            num_R=tcn_num_R,
            hidden_dim=tcn_hidden_dim,
            in_dim=self.model_dim,
            out_dim=self.output_channels,
        )
        self.rnn = GroupGRU(
            num_layers=rnn_num_layers,
            input_dim=self.model_dim,
            hidden_dim=rnn_hidden_dim,
        )
        self.res_scale = OutputScaling(dim=output_channels)

        # streaming component
        self.streaming_memory: List[ShortTermMemoryItem] = []
        self.prev_clip = None
    
    def forward(self, clip_embeds: Tensor):
        """
        Args:
            clip_embeds (Tensor): `[frame_cnt, patch_cnt_vertical, patch_cnt_horizontal, model_dim]`
        Returns:
            STMItem: Processed clip embeds
        """
        if self.disabled:
            printr(0, "STM disabled")
        
        prev_clip = self.prev_clip

        T, H, W, D  = clip_embeds.shape
        h, w        = int(H // self.pooling_scale_side), int(W // self.pooling_scale_side)
        printr(0, f"...stm original hw: {(H, W)}, new hw: {(h, w)}, drop ratio: {((1 - (h/H) * (w/W)) * 100):.2f}%")

        pooled_embeds = downsample_frames(clip_embeds, (h, w))      # -> [T, h, w, D]
        tempor_embeds = pooled_embeds.flatten(1, 2).transpose(0, 1) # -> [hw, T, D]

        tcn_input = tempor_embeds.transpose(-1, -2)  # -> [hw, D, T]
        # Concat a portion of frames from previous clip
        if prev_clip is not None:
            prev_clip = prev_clip.detach().transpose(-1, -2)[:, :, -clamp(prev_clip.shape[-1], 1, 10):]  # T_prev' = clamp(T_prev, 1, 10)
            if prev_clip.dim() == 2:
                prev_clip = prev_clip.unsqueeze(-1)
            tcn_input = torch.cat([prev_clip, tcn_input], dim=-1)  # -> [hw, D, T_prev'+T]
        self.prev_clip = tempor_embeds.detach()
        # Do TCN
        tcn_out = self.tcn(tcn_input)
        tcn_out = tcn_out[:, :, -T:]
        # Do RNN
        rnn_input   = tcn_out.transpose(-1, -2)  # -> [hw, T, D]
        rnn_out     = self.rnn(rnn_input, detach_h_prev=True)
        rnn_out     = rnn_input + rnn_out  # note: rnn_out is normed
        # Residual connection
        res_out = tempor_embeds.transpose(0, 1) + self.res_scale(rnn_out.transpose(0, 1), force_zero=self.disabled)  # -> [T, hw, D]

        self.streaming_memory.append(ShortTermMemoryItem(T, h, w, res_out.unsqueeze(0)))  # -> [1, T, hw, D]

        return self.streaming_memory[-1]
    
    def forward_old(self, clip_embeds: Tensor):
        """
        Args:
            clip_embeds (Tensor): `[frame_cnt, patch_cnt_vertical, patch_cnt_horizontal, model_dim]`
        Returns:
            STMItem: Processed clip embeds
        """
        if self.disabled:
            printr(0, "STM disabled")

        T, H, W, D = clip_embeds
        h, w = 8, 8

        # 把每一帧继续分 patch 并合并每个 patch
        merged_embs = downsample_frames(clip_embeds, (h, w))  # [T, h, w, D]

        # 不同帧中同位置的 patch 做 TCN
        patchwise_embs = merged_embs.flatten(1, 2).permute(1, 2, 0)  # [frm_cnt, #win*#win, d_model] -> [#win*#win, d_model, frm_cnt]
        tcn_input = patchwise_embs
        if len(self.streaming_memory) > 0:
            prev_clip = self.streaming_memory[-1].data.detach().squeeze(0).permute(1, 2, 0)  # [1, frm_cnt, #win*#win, d_model] -> [#win*#win, d_model, frm_cnt]
            prev_clip = prev_clip[:, :, -max(prev_clip.shape[2] // 2, 1):]              # 取用上一个片段的后一半帧当做历史
            if prev_clip.dim() == 2:
                prev_clip = prev_clip.unsqueeze(-1)
            tcn_input = torch.cat([prev_clip, patchwise_embs], dim=-1)  # [H*W, d_model, ++frm_cnt]
        tcn_out = self.tcn(tcn_input)                   # [#win*#win, d_model, ++frm_cnt]
        tcn_out = tcn_out[:, :, -T:]          # [#win*#win, d_model, frm_cnt]
        res_out = patchwise_embs.permute(2, 0, 1) + self.res_scale(tcn_out.permute(2, 0, 1), force_zero=self.disabled)  # [frm_cnt, #win*#win, d_model], residual conn
        self.streaming_memory.append(ShortTermMemoryItem(T, h, w, res_out.unsqueeze(0)))     # [1, frm_cnt, #win*#win, d_model]

        return self.streaming_memory[-1]
    
    @torch.no_grad()
    def get_frame_count(self):
        return sum(m.t for m in self.streaming_memory)
    
    @torch.no_grad()
    def truncate_streaming_memory(self):
        if "frame" in self.storage_method:
            if self.frame_storage < 0:
                return
            truncate_idx = len(self.streaming_memory)
            cu_frms_cnt = 0
            for mem in reversed(self.streaming_memory):
                truncate_idx -= 1
                cu_frms_cnt += mem.t
                if cu_frms_cnt >= self.frame_storage:
                    break
            for i in range(truncate_idx):
                self.streaming_memory[i] = None
        elif "clip" in self.storage_method:
            if self.clip_storage < 0:
                return
            for i in range(len(self.streaming_memory) - self.clip_storage):
                self.streaming_memory[i] = None
    
    def get_streaming_memory(self, as_one_tensor: bool = True):
        """
        Args:
            as_one_tensor (bool): if `True`, concat all the contents in `self.streaming_memory` into a single tensor; else, return the raw list of `self.streaming_memory`
        Returns:
            valid_stm_content (Tensor | List[STMItem]):
                `[1, stm_frm_cnt, stm_frm_patch_cnt, stm_output_channels]` or `self.streaming_memory`
        Note: calling this method does NOT automatically call streaming memory truncating method
        """
        # Valid STM content
        if as_one_tensor:
            return torch.cat([mem.data for mem in self.streaming_memory if mem is not None], dim=1)
        return self.streaming_memory

    @torch.no_grad()
    def clear_memory(self):
        self.rnn.clear_states()
        self.streaming_memory.clear()
        self.prev_clip = None


class LongTermHistoryQuery(nn.Module):
    def __init__(self, qcount: int = 32, hidden_dim: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.qcount = qcount
        self.queries = nn.Parameter(torch.randn(1, qcount, hidden_dim))
    
    def weight_init(self):
        init_scale = 0.02
        nn.init.normal_(self.queries.data, mean=0.0, std=init_scale * 0.7)
    
    def forward(self):
        return self.queries


class LongTermMemoryItem():
    def __init__(self, key: Tensor, data: Tensor):
        self.key = key
        self.data = data
        self.Nq = data.shape[1]

        ## Helper fields ##
        # Average over each clip's memory tokens ([B, Nq, D] -> [B, 1, D])
        self.key_cls = key.mean(dim=1, keepdim=True)
        self.data_cls = data.mean(dim=1, keepdim=True)


class LongTermMemory(nn.Module):
    def __init__(
            self,
            intra_frame_config: IntraFrameDecoderConfig = IntraFrameDecoderConfig(),
            inter_frame_config: InterFrameDecoderConfig = InterFrameDecoderConfig(),
            hist_hist_config:   InterClipDecoderConfig = InterClipDecoderConfig(),
            update_curr_config: InterClipDecoderConfig = InterClipDecoderConfig(),
            update_hist_config: InterClipDecoderConfig = InterClipDecoderConfig(),
            hist_selection_method: Literal["last", "topk", "prob"] = "topk",
            sim_scores_alpha: float = 0.35,
            sim_scores_threshold: float = 0.7,
            sim_scores_topk: int = 5,
            # graphbuild_head_num: int = 1,
            graphbuild_hidden_dim: int = 512,
            graphbuild_mlp_nlayers: int = 2,
            graphbuild_sim_thresholds: Tuple[float] = (0.85, 0.9, 0.96, 0.98),  # TODO: ready to change these values
            **kwargs
    ):
        super().__init__(**kwargs)

        self.training_with_eval = int(os.environ.get("O2O_TRAINING", 0))

        self.cfg_intraf = intra_frame_config
        self.cfg_interf = inter_frame_config
        self.cfg_hishis = hist_hist_config
        self.cfg_updcur = update_curr_config
        self.cfg_updhis = update_hist_config

        self.disabled = False

        self.queries_temporal = LongTermHistoryQuery(self.cfg_interf.num_queries, self.cfg_interf.d_model)

        self.decoder_spatial = MemoryAttentionDecoder(
            num_layers=self.cfg_intraf.num_layers,
            d_model=self.cfg_intraf.d_model,
            num_heads=self.cfg_intraf.num_heads,
            ff_dim=self.cfg_intraf.ff_dim,
            attn_dropout=self.cfg_intraf.attn_dropout,
            ffn_dropout=self.cfg_intraf.ffn_dropout,
            use_qkvo_proj=self.cfg_intraf.use_qkvo_proj,
            only_residual_last_layer=self.cfg_intraf.only_residual_last_layer,
            query_residual_scale=self.cfg_intraf.query_residual_scale,
            value_residual_scale=self.cfg_intraf.value_residual_scale,
            ffn_residual_scale=self.cfg_intraf.ffn_residual_scale,
        )
        self.decoder_temporal = MemoryAttentionDecoder(
            num_layers=self.cfg_interf.num_layers,
            d_model=self.cfg_interf.d_model,
            num_heads=self.cfg_interf.num_heads,
            ff_dim=self.cfg_interf.ff_dim,
            attn_dropout=self.cfg_interf.attn_dropout,
            ffn_dropout=self.cfg_interf.ffn_dropout,
            use_qkvo_proj=self.cfg_interf.use_qkvo_proj,
            only_residual_last_layer=self.cfg_interf.only_residual_last_layer,
            query_residual_scale=self.cfg_interf.query_residual_scale,
            value_residual_scale=self.cfg_interf.value_residual_scale,
            ffn_residual_scale=self.cfg_interf.ffn_residual_scale,
        )
        self.decoders_hist_to_hist = nn.ModuleList([
            MemoryAttentionDecoder(
                num_layers=self.cfg_hishis.num_layers,
                d_model=self.cfg_hishis.d_model,
                num_heads=self.cfg_hishis.num_heads,
                ff_dim=self.cfg_hishis.ff_dim,
                attn_dropout=self.cfg_hishis.attn_dropout,
                ffn_dropout=self.cfg_hishis.ffn_dropout,
                use_qkvo_proj=self.cfg_hishis.use_qkvo_proj,
                only_residual_last_layer=self.cfg_hishis.only_residual_last_layer,
                query_residual_scale=self.cfg_hishis.query_residual_scale,
                value_residual_scale=self.cfg_hishis.value_residual_scale,
                ffn_residual_scale=self.cfg_hishis.ffn_residual_scale,
                attn_implementation=GraphLikeMultiheadAttention,
            ) for _ in range(len(graphbuild_sim_thresholds))
        ])
        self.decoder_hist_to_curr = MemoryAttentionDecoder(
            num_layers=self.cfg_updcur.num_layers,
            d_model=self.cfg_updcur.d_model,
            num_heads=self.cfg_updcur.num_heads,
            ff_dim=self.cfg_updcur.ff_dim,
            attn_dropout=self.cfg_updcur.attn_dropout,
            ffn_dropout=self.cfg_updcur.ffn_dropout,
            use_qkvo_proj=self.cfg_updcur.use_qkvo_proj,
            only_residual_last_layer=self.cfg_updcur.only_residual_last_layer,
            query_residual_scale=self.cfg_updcur.query_residual_scale,
            value_residual_scale=self.cfg_updcur.value_residual_scale,
            ffn_residual_scale=self.cfg_updcur.ffn_residual_scale,
            k_residual=True,
        )
        self.decoder_curr_to_hist = MemoryAttentionDecoder(
            num_layers=self.cfg_updhis.num_layers,
            d_model=self.cfg_updhis.d_model,
            num_heads=self.cfg_updhis.num_heads,
            ff_dim=self.cfg_updhis.ff_dim,
            attn_dropout=self.cfg_updhis.attn_dropout,
            ffn_dropout=self.cfg_updhis.ffn_dropout,
            use_qkvo_proj=self.cfg_updhis.use_qkvo_proj,
            only_residual_last_layer=self.cfg_updhis.only_residual_last_layer,
            query_residual_scale=self.cfg_updhis.query_residual_scale,
            value_residual_scale=self.cfg_updhis.value_residual_scale,
            ffn_residual_scale=self.cfg_updhis.ffn_residual_scale,
        )
        # self.decoder_history = MemoryDecoder(
        #     self.cfg_updcur.num_layers, self.cfg_updcur.d_model, self.cfg_updcur.num_heads,
        #     self.cfg_updcur.ff_dim, self.cfg_updcur.attn_dropout, self.cfg_updcur.ffn_dropout,
        # )

        # self.out_scale = OutputScaling(dim=self.cfg_intraf.d_model)

        self.gnode_norm = nn.LayerNorm(self.cfg_intraf.d_model)
        self.gnode_proj = MLPProj(self.cfg_intraf.d_model, graphbuild_hidden_dim, self.cfg_intraf.d_model, graphbuild_mlp_nlayers)
        # self.gnode_proj = nn.Linear(self.cfg_intraf.d_model, self.cfg_intraf.d_model)

        self.streaming_memory: List[LongTermMemoryItem] = []

        self.hist_selection_method = hist_selection_method

        # Similarity scores by moving average
        self.sim_scores = []
        self.sim_alpha = sim_scores_alpha
        self.sim_threshold = sim_scores_threshold
        self.sim_topk = sim_scores_topk

        self.gb_head_num = self.cfg_hishis.num_heads
        self.gb_sim_thresholds = graphbuild_sim_thresholds
    
    def weight_init(self):
        nn.init.ones_(self.gnode_norm.weight)
        if hasattr(self.gnode_norm, 'bias') and self.gnode_norm.bias is not None:
            nn.init.zeros_(self.gnode_norm.bias)

    def build_clip_graph(self, sim_threshold: float = 0.7, thresholding_sharpness: float = 8.0, no_self_loop: bool = True, clip_repr: Optional[Tensor] = None):
        """
        Args:
            clip_repr: shaped `[B, N, D]`
        
        Returns:
            TensorTuple ([bool tensor, float tensor]):
            adjacency_matrix_hard (BoolTensor): shaped `[B, H, Nc*Nq, Nc*Nq]` where `Nc` is clips count and `Nq` is token count for each clip memory; None if no memory.
                This one maintains no gradient and should only be used as masks e.g. attn masks.
            
            adhacency_matrix_soft (FloatTensor): shaped `[B, H, Nc*Nq, Nc*Nq]`. a softer version of adjacency matrix. <= 0.5 means unconnected, > 0.5 means connected; None if no memory.
                This one uses sigmoid to assign smooth weights to graph's edges. Control the sharpness of weight assignment via parameter.
        """
        B, Nq, D = self.streaming_memory[-1].data.shape
        H = self.gb_head_num
        d = D // self.gb_head_num
        assert H * d == D
        
        if clip_repr is None:
            # Use streaming memory
            if not self.streaming_memory or len(self.streaming_memory) <= 0:
                return None, None

            mems_cls = [mem.key_cls for mem in self.streaming_memory]
            mems_cls = torch.cat(mems_cls, dim=1).detach()  # [B, Nc, D], DETACH, no gradient flowing back to memory tokens before mean pooling
        else:
            # Use the given clip representation
            mems_cls = clip_repr.clone()
        
        projed_cls = self.gnode_proj(self.gnode_norm(mems_cls)) + mems_cls
        headed_cls = projed_cls.contiguous().view(B, -1, H, d).transpose(1, 2)  # [B, H, Nc, d]

        normed_cls = F.normalize(headed_cls, p=2, dim=-1)
        cos_sim = normed_cls @ normed_cls.transpose(-1, -2)  # [B, H, Nc, d] @ [B, H, d, Nc] -> [B, H, Nc, Nc]
        if no_self_loop:
            cos_sim = cos_sim.masked_fill(torch.eye(cos_sim.shape[-1], dtype=torch.bool, device=cos_sim.device), float("-inf"))

        adjacent_mat_hard = cos_sim >= sim_threshold
        adjacent_mat_soft = torch.sigmoid((cos_sim - sim_threshold) * thresholding_sharpness)
        
        # Expand [Nc, Nc] to [Nc*Nq, Nc*Nq]
        adjacent_blkmat_hard = adjacent_mat_hard.repeat_interleave(Nq, dim=-1).repeat_interleave(Nq, dim=-2)
        adjacent_blkmat_soft = adjacent_mat_soft.repeat_interleave(Nq, dim=-1).repeat_interleave(Nq, dim=-2)

        return adjacent_blkmat_hard, adjacent_blkmat_soft
    
    def forward(self, clip_embeds: Tensor, interaction_callback: Optional[Callable[[Tensor], None]] = None, **interaction_args):
        """
        Args:
            clip_embeds (Tensor): `[1, frame_cnt, stm_patch_cnt, stm_out_channels]`
        Returns:
            LTMItem: Processed clip embeds
        """
        if self.disabled:
            printr(0, "LTM disabled")
        
        batch_size, frame_count, patch_count, channels = tuple(clip_embeds.shape)

        ## Intra-frame query
        frames_embeds = clip_embeds.reshape(batch_size * frame_count, patch_count, channels)
        frames_embeds_attn = self.decoder_spatial(frames_embeds, frames_embeds)     # intra-frame self-attn
        frames_embeds_attn = frames_embeds + frames_embeds_attn                     # [1*frm_cnt, patch_cnt, d_model], resid conn  TODO: 这一块可以替换成 decoder_spatial 内部的 query_residual
        frames_embeds_attn = frames_embeds_attn.reshape(batch_size, frame_count, -1, channels)  # [1, frm_cnt, patch_cnt, d_model]

        ## Inter-frame query
        h = frames_embeds_attn.flatten(1, 2)     # [1, frm_cnt*patch_cnt, d_model]
        h_key = self.decoder_temporal(self.queries_temporal().expand(batch_size, -1, -1), h)    # [1, Nq_t, d_model]
        h_val = h_key.clone()
        h_key = h_key.detach()

        ## Inter-clip attend
        if len(self.streaming_memory) > 0:
            # Select clip memory...
            if self.training_with_eval or self.training or "last" in self.hist_selection_method.lower():
                # ...use hard-coded last-2 method during train time (to avoid dead-locks)
                printr(0, "...ltm using last 2 as history")
                hist_indices = torch.arange(len(self.streaming_memory))
                hist_indices = hist_indices[-min(2, len(self.streaming_memory)):]
            elif "top" in self.hist_selection_method.lower():
                # ...use similarity method during inference time
                printr(0, "...ltm using selective history")
                # Similarity scores
                sim_scores_new = [F.cosine_similarity(mem.key_cls.squeeze(1), h_key.mean(dim=1), dim=-1).item() for mem in self.streaming_memory]  # [1, d_model] -> [1] x memory clips count
                # Moving average
                self.sim_scores.append(sim_scores_new[-1])  # Before appending, len(self.sim_scores) == len(sim_scores_new) - 1
                sim_scores_old  = torch.tensor(self.sim_scores, device=h_val.device)
                sim_scores_new  = torch.tensor(sim_scores_new, device=h_val.device)
                sim_scores      = self.sim_alpha * sim_scores_new + (1 - self.sim_alpha) * sim_scores_old
                hist_indices    = torch.where(sim_scores > self.sim_threshold)[0]
                # Apply top-k constraint
                if len(hist_indices) > self.sim_topk:
                    _, hist_indices = torch.topk(sim_scores, self.sim_topk)  # TODO: 排一下序，按照时间顺序来
            elif "prob" in self.hist_selection_method.lower():
                pass  # TODO
            else:
                raise ValueError(f"LTM history clip selection method \"{self.hist_selection_method}\" is not defined")
            
            # Call for interaction with STM
            if interaction_callback is not None and hist_indices.numel() > 0:
                interaction_callback(hist_indices, **interaction_args)
            
            # Two-way fusing
            if hist_indices.numel() > 0:
                N_sel, N_q  = len(hist_indices), self.streaming_memory[-1].data.shape[1]
                N_mem       = len(self.streaming_memory)
                # History <-> history
                mem         = [mem.data for mem in self.streaming_memory]  # [B=1, L=Nq, D] x Nmem
                mem_seq     = torch.cat(mem, dim=1)  # [1, Nmem*Nq, D]
                mem_new     = mem_seq.detach()  # DETACH original history here
                debug_avg_deg = []
                for h2h, graph_th in zip(self.decoders_hist_to_hist, self.gb_sim_thresholds):
                    mem_new_cls = mem_new.reshape(mem_new.shape[0], N_mem, N_q, -1).mean(dim=-2)  # -> [1, Nmem, Nq, D] -> [1, Nmem, D]
                    adj_mat, conn_mat = self.build_clip_graph(
                        sim_threshold=graph_th, thresholding_sharpness=8.0, no_self_loop=True,
                        clip_repr=mem_new_cls
                    )  # TODO: eliminate these magic numbers
                    debug_avg_deg.append((torch.round(adj_mat.sum(-1).to(torch.float).mean() / N_q * 100).item() / 100))
                    mem_new = h2h(
                        mem_new, mem_new,
                        adjacency_mat=adj_mat, connectivity_mat=conn_mat
                    )  # query residual equipped
                printr(0, f"...graph node count: {N_mem}; avg degrees: " + ", ".join([f"threshold {th} - {deg}" for th, deg in zip(self.gb_sim_thresholds, debug_avg_deg)]))
                # Two copies of interacted memory, and original memory
                mem_old     = mem_seq.contiguous().view(mem_seq.shape[0], N_mem, N_q, -1)
                mem_new     = mem_new.contiguous().view(mem_new.shape[0], N_mem, N_q, -1)
                hist_old    = mem_old[:, hist_indices, :, :]  # [1, Nsel, Nq, D]
                hist_new    = mem_new[:, hist_indices, :, :]  # [1, Nsel, Nq, D]
                # History -> current; use interacted history
                hist_new_1      = hist_new.flatten(1, 2)  # no detaching
                h_val           = self.decoder_hist_to_curr(h_val, hist_new_1)  # NOTE: K/V residual applied here
                # Current -> history; use original history
                hist_old_1      = hist_old.squeeze(0)  # [N_sel, N_q, d_model]
                h_key_copy      = h_key.expand(hist_old_1.shape[0], -1, -1).detach()  # DETACH current memory
                enhanced_hist   = self.decoder_curr_to_hist(hist_old_1, h_key_copy)
                for i, idx in enumerate(hist_indices):
                    self.streaming_memory[idx].data = enhanced_hist[i].unsqueeze(0)  # [1, N_q, d_model]
        
        self.streaming_memory.append(LongTermMemoryItem(key=h_key, data=h_val))  # [1, Nq_t, d_model]
        return self.streaming_memory[-1]

    @torch.no_grad()
    def forward_streaming(self, video_embeds: torch.Tensor):
        self.forward([video_embeds])
    
    @torch.no_grad()
    def truncate_streaming_memory(self):
        """Saving only the last video clip's memory"""
        self.streaming_memory = self.streaming_memory[-1:]
    
    def get_streaming_memory(self, as_one_tensor: bool = True):
        """
        Returns:
            valid_ltm_content (Tensor | list[LTMItem]): `[1, ltm_seq_len, stm_output_channels]` or `self.streaming_memory`
        """
        if as_one_tensor:
            return torch.cat([m.data for m in self.streaming_memory], dim=1)
        return self.streaming_memory

    @torch.no_grad()
    def clear_memory(self):
        self.streaming_memory.clear()
        self.sim_scores.clear()


class ShortLongModulate(nn.Module):
    """Long-to-short Modulate"""
    def __init__(self, model_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.attn = MultiheadAttention(
            embed_dim=model_dim,
            num_heads=8,
            q_dim=model_dim,
            use_q_proj=True,  # Wq, Wk: Align the modality of STM query and LTM key
            use_k_proj=True,
            use_v_proj=False,  # !Wv, !Wo: Use original LTM modality to get gamma and beta 
            use_out_proj=False,
        )
        self.proj_gamma = nn.Linear(model_dim, model_dim)
        self.proj_beta = nn.Linear(model_dim, model_dim)
        self.scale_gamma = OutputScaling(init_value=1e-3, init_bias=1.0, dim=model_dim)
        self.scale_beta = OutputScaling(init_value=1e-3, dim=model_dim)
    
    def weight_init(self):
        for module in (self.proj_gamma, self.proj_beta):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, stm: Tensor, ltm: Tensor):
        """
        Args:
            stm: shaped `[B, T, HW, D]`
            ltm: shaped `[B, k*Nq, D]`
        Returns:
            new_stm: shaped `[B, T, HW, D]`
        """
        s = stm.flatten(1, 2).mean(1, keepdim=True)  # -> [B, 1, D]
        l, _ = self.attn(s, ltm, ltm)  # -> [B, 1, D]

        gamma = self.scale_gamma(self.proj_gamma(l))
        beta = self.scale_beta(self.proj_beta(l))

        # Batchwise calc
        stm_new = []
        for i, ss in enumerate(stm):
            ss = ss * gamma[i] + beta[i]
            stm_new.append(ss)
        stm_new = torch.stack(stm_new, dim=0)
        return stm_new


class ShortLongAttend(nn.Module):
    def __init__(self, model_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.attn = MultiheadAttention(
            embed_dim=model_dim,
            num_heads=8,
            q_dim=model_dim,
            use_q_proj=True,
            use_k_proj=True,
            use_v_proj=True,
            use_out_proj=True,
        )
        self.long_scale = OutputScaling(init_value=0.01, dim=model_dim)
    
    def forward(self, stm: Tensor, ltm: Tensor):
        """
        Args:
            stm: shaped `[B, T, HW, D]`
            ltm: shaped `[B, k*Nq, D]`
        Returns:
            new_stm: shaped `[B, T, HW, D]`
        """
        B, T, HW, D = stm.shape

        s = stm.flatten(1, 2)  # -> [B, THW, D]
        l, _ = self.attn(q=s, k=ltm, v=ltm)  # -> [B, THW, D]
        
        stm_new = s + self.long_scale(l)
        return stm_new.view(B, T, HW, D)


class Memories(nn.Module):
    def __init__(
            self,
            stm_storage: int = 2,  # 👆 -1 is used when testing forward-active-responding-like tasks
            stm_storage_method: Literal['clip', 'frame'] = 'clip',  # 👆 训练时用 clip 方法，测试时用 frame 方法
            memory_zip_method: Literal['interleave', 'ltm_overwrites_stm', 'stm_overwrites_ltm'] = 'interleave',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.stm = ShortTermMemory(
            storage=stm_storage,
            pooling_scale_side=2,  # NOTE: set to 1.33 if evaluating StreamingBench
            model_dim=3584,
            tcn_num_PG_layers=2,
            tcn_num_R_layers=1,
            tcn_num_R=3,
            tcn_hidden_dim=512,
            rnn_num_layers=3,
            rnn_hidden_dim=512,
            output_channels=3584,
            storage_method=stm_storage_method,
        )
        self.ltm = LongTermMemory(
            intra_frame_config=IntraFrameDecoderConfig(  # Q: each frame, K/V: each frame
                num_layers=2, d_model=3584, num_heads=16, ff_dim=4096, attn_dropout=0.1, ffn_dropout=0.1,
                query_residual_scale=None, value_residual_scale=None, ffn_residual_scale=None,
            ),
            inter_frame_config=InterFrameDecoderConfig(  # Q: learnable queries, K/V: clip frames
                num_queries=64, num_layers=3, d_model=3584, num_heads=16, ff_dim=4096, attn_dropout=0.1, ffn_dropout=0.1,
                query_residual_scale=None, value_residual_scale=1, ffn_residual_scale=1, only_residual_last_layer=True,
            ),
            hist_hist_config=InterClipDecoderConfig(
                num_layers=1, d_model=3584, num_heads=16, ff_dim=4096, attn_dropout=0.1, ffn_dropout=0.1,
                query_residual_scale=0.9, value_residual_scale=None, ffn_residual_scale=0.1, only_residual_last_layer=True,
                use_qkvo_proj=(True, True, True, True),
            ),
            update_curr_config=InterClipDecoderConfig(  # Q: curr clip queries, K/V: hist clip queries
                num_layers=3, d_model=3584, num_heads=16, ff_dim=4096, attn_dropout=0.1, ffn_dropout=0.1,
                query_residual_scale=0.9, value_residual_scale=None, ffn_residual_scale=0.1, only_residual_last_layer=True,
            ),
            update_hist_config=InterClipDecoderConfig(  # Q: hist clip queries, K/V: curr clip queries
                num_layers=2, d_model=3584, num_heads=16, ff_dim=4096, attn_dropout=0.1, ffn_dropout=0.1,
                query_residual_scale=0.9, value_residual_scale=None, ffn_residual_scale=0.1, only_residual_last_layer=True,
            ),
            hist_selection_method="topk",
            sim_scores_alpha=0.35,
            sim_scores_threshold=0.7,
            sim_scores_topk=5,
            graphbuild_hidden_dim=512,
            graphbuild_mlp_nlayers=2,
            graphbuild_sim_thresholds=(
                (0.85, 0.9, 0.96, 0.98) if self.training or self.ltm.training_with_eval else (0.98, 0.96, 0.9, 0.85)
            ),  # NOTE: 0.85 to 0.98 while training, 0.98 to 0.85 while evaluating
        )
        self.sl_interact = ShortLongModulate(
            model_dim=3584,
        )
        # self.sl_interact = ShortLongAttend(
        #     model_dim=3584,
        # )

        self.memory_zip_method = memory_zip_method

        self.token_cnt_before_drops = []
        self.token_cnt_after_drops = []

        for name, module in self.named_modules():
            module.register_forward_hook(forward_anomaly_hook(name))
        
        for name, param in self.ltm.gnode_proj.named_parameters():
            if "0" in name:
                param.register_hook(backward_grad_hook(f"ltm.gnode_proj.{name}"))
        
        # for name, param in self.ltm.named_parameters():
        #     if "attn" in name:
        #         param.register_hook(backward_grad_hook(f"ltm.decoder_hist_to_hist.{name}"))
        
        self.gnode_proj_weight_cache = None
        self.decoder_h2h_weight_cache = None
    
    def short_long_interaction(self, ltm_indices: Tensor, clip_id: int, **kwargs):
        printr(0, "...stm & ltm interaction")
        # Get STM & LTM
        selected_ltm = [self.ltm.streaming_memory[idx] for idx in ltm_indices]
        ltm = torch.cat([mem.data.detach() for mem in selected_ltm], dim=1)  # -> [B, k*Nq, D]; DETACH
        stm = self.stm.streaming_memory[clip_id].data  # -> [B, T, hw, D]
        # Interacting
        stm_new = self.sl_interact.forward(stm, ltm)  # -> [B, T, hw, D]
        self.stm.streaming_memory[clip_id].data = stm_new
    
    def process_videos_embeddings(self, reshaped_videos_embeds: Union[torch.Tensor, list[torch.Tensor]]):
        """
        Args:
            reshaped_videos_embeds (Tensor | list[Tensor]): `[frame_cnt, patch_cnt_v, patch_cnt_h, model_dim]` x `video_cnt` or `1`
        Returns:
            `(STM mem, LTM mem, Mem grid THW, STM scatter lengths, LTM scatter lengths)`, STM/LTM mem shaped like `[1, vid_seq_len, model_dim]`. 使用 grid THW 的时候记得调整其 device, dtype, requires_grad 属性
        """
        printr(0, f"v2.4.3-ltm-all-sattn-prebuildgraph-ngranu")
        printr(0, f"{self.sl_interact.scale_gamma.a.data=},\n{self.sl_interact.scale_gamma.b.data=}")
        printr(0, f"gnode_proj weight updated: {(self.ltm.gnode_proj[2].weight.clone().detach() != self.gnode_proj_weight_cache).any().item() if self.gnode_proj_weight_cache is not None else None}")
        printr(0, f"decoder h2h qproj weight updated: {(self.ltm.decoders_hist_to_hist[0].layers[0].attn.q_proj.weight.clone().detach() != self.decoder_h2h_weight_cache).any().item() if self.decoder_h2h_weight_cache is not None else None}")
        self.gnode_proj_weight_cache = self.ltm.gnode_proj[2].weight.clone().detach()
        self.decoder_h2h_weight_cache = self.ltm.decoders_hist_to_hist[0].layers[0].attn.q_proj.weight.clone().detach()

        if isinstance(reshaped_videos_embeds, torch.Tensor):  # pack into list
            reshaped_videos_embeds = [reshaped_videos_embeds]
        
        for i, clip_embeds in enumerate(reshaped_videos_embeds):
            stm_item = self.stm.forward(clip_embeds)
            ltm_item = self.ltm.forward(stm_item.data.detach(), interaction_callback=self.short_long_interaction, clip_id=i)  # DETACH

        self.stm.truncate_streaming_memory()

    def prepare_input(
            self,
            input_ids_segments: List[Tensor],
            inputs_segments: List[Tensor],
            labels_segments: Optional[List[Tensor]],
            input_ids_old: Tensor,
            inputs_embeds_old: Tensor,
            labels_old: Optional[Tensor],
            position_ids_old: Optional[Tensor] = None,
            attention_mask_old: Optional[Tensor] = None,
            video_grid_thw_old: Optional[Tensor] = None,
            second_per_grid_ts_old: Optional[Tensor] = None,
            ignore_label: Optional[int] = -100,
            video_pad_token: Optional[int] = None,
            vision_start_token: Optional[int] = None,
            vision_end_token: Optional[int] = None,
            vision_start_embed: Optional[Tensor] = None,
            vision_end_embed: Optional[Tensor] = None,
            spatial_merge_size: Optional[int] = None,
    ):
        """
        Args:
            vision_start_embed: shape `[1, D]`
            vision_emd_embed: shape `[1, D]`
        Returns:
            `input_ids_new`, `inputs_embeds_new`, `labels_new`, `position_ids_new` (None), `attention_mask_new`, `video_grid_thw_new`, `second_per_grid_ts_new`
        """
        from .mm_utils import _cat_mask_values

        # Get memory list
        stm_mem = self.stm.get_streaming_memory(as_one_tensor=False)    # [1, frms, stm_frm_patch_cnt, d_model] x stm_clip_cnt
        ltm_mem = self.ltm.get_streaming_memory(as_one_tensor=False)    # [1, frms, d_model] x ltm_clip_cnt
        printr(0, f"STM clip count: {len(stm_mem)}, LTM clip count: {len(ltm_mem)}")
        # Extend memory list to the same length
        mem_len = max(len(stm_mem), len(ltm_mem))
        stm_mem = [None] * (mem_len - len(stm_mem)) + stm_mem   # Prepend None's
        ltm_mem += [None] * (mem_len - len(ltm_mem))            # Append None's

        # Prepare new inputs
        input_ids_new       = torch.empty(0, device=input_ids_old.device, dtype=input_ids_old.dtype, requires_grad=input_ids_old.requires_grad)
        inputs_embeds_new   = torch.empty(0, device=inputs_embeds_old.device, dtype=inputs_embeds_old.dtype, requires_grad=inputs_embeds_old.requires_grad)
        labels_new          = torch.empty(0, device=labels_old.device, dtype=labels_old.dtype, requires_grad=labels_old.requires_grad) if labels_old is not None else None
        if labels_segments is None:
            labels_segments = [None] * len(input_ids_segments)
        assert len(input_ids_segments) == len(inputs_segments) == len(labels_segments)

        # Construct new inputs
        B, N, D = inputs_embeds_old.shape
        clip_mem_id = 0
        for idseg, emseg, lbseg in zip(input_ids_segments, inputs_segments, labels_segments):
            label_available = lbseg is not None and labels_new is not None

            # Concat original text embeds
            input_ids_new       = torch.cat([input_ids_new, idseg.unsqueeze(0)], dim=1)
            inputs_embeds_new   = torch.cat([inputs_embeds_new, emseg.unsqueeze(0)], dim=1)
            labels_new          = torch.cat([labels_new, lbseg.unsqueeze(0)], dim=1) if label_available else None

            # Concat memory embeds
            if clip_mem_id < mem_len:
                lmem, smem = ltm_mem[clip_mem_id], stm_mem[clip_mem_id]
                lmem, smem = None if lmem is None else lmem.data, None if smem is None else smem.data
                # Zip STM and LTM
                if "inter" in self.memory_zip_method.lower():
                    # Concat LTM
                    if lmem is not None:
                        input_ids_new       = _cat_mask_values(input_ids_new, video_pad_token, lmem.shape[1], has_embed_dim=False, batchsize=B)
                        inputs_embeds_new   = torch.cat([inputs_embeds_new, lmem], dim=1)
                        labels_new          = _cat_mask_values(labels_new, ignore_label, lmem.shape[1], has_embed_dim=False, batchsize=B) if label_available else None
                    # Concat separator tokens (for qwen, they are `<|vision_end|>` and `<|vision_start|>`)
                    if lmem is not None and smem is not None:
                        input_ids_new = _cat_mask_values(input_ids_new, vision_end_token, 1, has_embed_dim=False, batchsize=B)      # ltm ends
                        input_ids_new = _cat_mask_values(input_ids_new, vision_start_token, 1, has_embed_dim=False, batchsize=B)    # stm starts
                        inputs_embeds_new = torch.cat([inputs_embeds_new, vision_end_embed.unsqueeze(0)], dim=1)
                        inputs_embeds_new = torch.cat([inputs_embeds_new, vision_start_embed.unsqueeze(0)], dim=1)
                        labels_new = _cat_mask_values(labels_new, ignore_label, 2, has_embed_dim=False, batchsize=B) if label_available else None
                    # Concat STM
                    if smem is not None:
                        smem = smem.flatten(1, 2)
                        input_ids_new       = _cat_mask_values(input_ids_new, video_pad_token, smem.shape[1], has_embed_dim=False, batchsize=B)
                        inputs_embeds_new   = torch.cat([inputs_embeds_new, smem], dim=1)
                        labels_new          = _cat_mask_values(labels_new, ignore_label, smem.shape[1], has_embed_dim=False, batchsize=B) if label_available else None
                elif "stm_over" in self.memory_zip_method.lower():
                    # Concat STM
                    if smem is not None:
                        smem = smem.flatten(1, 2)
                        input_ids_new       = _cat_mask_values(input_ids_new, video_pad_token, smem.shape[1], has_embed_dim=False, batchsize=B)
                        inputs_embeds_new   = torch.cat([inputs_embeds_new, smem], dim=1)
                        labels_new          = _cat_mask_values(labels_new, ignore_label, smem.shape[1], has_embed_dim=False, batchsize=B) if label_available else None
                    # Concat LTM if STM does not exist
                    elif lmem is not None:
                        input_ids_new       = _cat_mask_values(input_ids_new, video_pad_token, lmem.shape[1], has_embed_dim=False, batchsize=B)
                        inputs_embeds_new   = torch.cat([inputs_embeds_new, lmem], dim=1)
                        labels_new          = _cat_mask_values(labels_new, ignore_label, lmem.shape[1], has_embed_dim=False, batchsize=B) if label_available else None
                elif "ltm_over" in self.memory_zip_method.lower():
                    # Concat LTM
                    if lmem is not None:
                        input_ids_new       = _cat_mask_values(input_ids_new, video_pad_token, lmem.shape[1], has_embed_dim=False, batchsize=B)
                        inputs_embeds_new   = torch.cat([inputs_embeds_new, lmem], dim=1)
                        labels_new          = _cat_mask_values(labels_new, ignore_label, lmem.shape[1], has_embed_dim=False, batchsize=B) if label_available else None
                    # Concat STM if LTM does not exist
                    elif smem is not None:
                        smem = smem.flatten(1, 2)
                        input_ids_new       = _cat_mask_values(input_ids_new, video_pad_token, smem.shape[1], has_embed_dim=False, batchsize=B)
                        inputs_embeds_new   = torch.cat([inputs_embeds_new, smem], dim=1)
                        labels_new          = _cat_mask_values(labels_new, ignore_label, smem.shape[1], has_embed_dim=False, batchsize=B) if label_available else None
                else:
                    raise ValueError(f"Memory zip method {self.memory_zip_method} does not exist")
                clip_mem_id += 1
        
        # Remaining new input components (posidx, attnmask)
        position_ids_new = None
        attention_mask_new = torch.ones((B, inputs_embeds_new.shape[1]), device=attention_mask_old.device, dtype=attention_mask_old.dtype, requires_grad=attention_mask_old.requires_grad)

        # Additional [video grid THW] and [second per grid] for qwen
        mem_grid_thw = []
        second_per_grid_ts_new = []
        for clipid, (smem, lmem) in enumerate(zip(stm_mem, ltm_mem)):
            if lmem is None and smem is None:  # a pair of vision tokens containing nothing (i.e. only `<|vision_start|><|vision_end|>`)
                mem_grid_thw.append([0, 0, 0])
                second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
            elif "inter" in self.memory_zip_method.lower():
                if lmem is not None:
                    mem_grid_thw.append([lmem.Nq, 1 * spatial_merge_size, 1 * spatial_merge_size])
                    second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
                if smem is not None:
                    mem_grid_thw.append([smem.t, smem.h * spatial_merge_size, smem.w * spatial_merge_size])
                    second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
            elif "stm_over" in self.memory_zip_method.lower():
                if smem is not None:
                    mem_grid_thw.append([smem.t, smem.h * spatial_merge_size, smem.w * spatial_merge_size])
                    second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
                elif lmem is not None:
                    mem_grid_thw.append([lmem.Nq, 1 * spatial_merge_size, 1 * spatial_merge_size])
                    second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
                else:
                    raise NotImplementedError()
            elif "ltm_over" in self.memory_zip_method.lower():
                if lmem is not None:
                    mem_grid_thw.append([lmem.Nq, 1 * spatial_merge_size, 1 * spatial_merge_size])
                    second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
                elif smem is not None:
                    mem_grid_thw.append([smem.t, smem.h * spatial_merge_size, smem.w * spatial_merge_size])
                    second_per_grid_ts_new.append(second_per_grid_ts_old[clipid])
                else:
                    raise NotImplementedError()
            else:
                raise ValueError(f"Memory zip method {self.memory_zip_method} does not exist")
        mem_grid_thw = torch.tensor(mem_grid_thw, dtype=video_grid_thw_old.dtype, device=video_grid_thw_old.device, requires_grad=video_grid_thw_old.requires_grad)

        # Save statistic data
        self.token_cnt_before_drops.append(inputs_embeds_old.shape[1])
        self.token_cnt_after_drops.append(inputs_embeds_new.shape[1])

        return input_ids_new, inputs_embeds_new, labels_new, position_ids_new, attention_mask_new, mem_grid_thw, second_per_grid_ts_new
    
    def prepare_input_only_visual(self, device = None, dtype = None, requires_grad = None):
        """
        Returns:
            memory_features (Tensor): shaped `[B=1, L=flattened memory token count, D=3584]`
        """
        # Get memory list
        stm_mem = self.stm.get_streaming_memory(as_one_tensor=False)    # [1, frms, stm_frm_patch_cnt, d_model] x stm_clip_cnt
        ltm_mem = self.ltm.get_streaming_memory(as_one_tensor=False)    # [1, frms, d_model] x ltm_clip_cnt
        printr(0, f"STM clip count: {len(stm_mem)}, LTM clip count: {len(ltm_mem)}")
        # Extend memory list to the same length
        mem_len = max(len(stm_mem), len(ltm_mem))
        stm_mem = [None] * (mem_len - len(stm_mem)) + stm_mem   # Prepend None's
        ltm_mem += [None] * (mem_len - len(ltm_mem))            # Append None's

        # Prepare new inputs
        mem_embeds = torch.empty(0, device=device, dtype=dtype, requires_grad=requires_grad)

        # Concat memory embeds
        for lmem, smem in zip(ltm_mem, stm_mem):
            lmem, smem = None if lmem is None else lmem.data, None if smem is None else smem.data
            # Zip STM and LTM
            if "inter" in self.memory_zip_method.lower():
                # Concat LTM
                if lmem is not None:
                    mem_embeds = mem_embeds.to(dtype=lmem.dtype, device=lmem.device)
                    mem_embeds = torch.cat([mem_embeds, lmem], dim=1)
                # Concat separator tokens (for qwen, they are `<|vision_end|>` and `<|vision_start|>`)
                if lmem is not None and smem is not None:
                    # mem_embeds = torch.cat([mem_embeds, vision_end_embed.unsqueeze(0)], dim=1)
                    # mem_embeds = torch.cat([mem_embeds, vision_start_embed.unsqueeze(0)], dim=1)
                    pass
                # Concat STM
                if smem is not None:
                    smem = smem.flatten(1, 2)
                    mem_embeds = mem_embeds.to(dtype=smem.dtype, device=smem.device)
                    mem_embeds = torch.cat([mem_embeds, smem], dim=1)
            elif "stm_over" in self.memory_zip_method.lower():
                # Concat STM
                if smem is not None:
                    smem = smem.flatten(1, 2)
                    mem_embeds = mem_embeds.to(dtype=smem.dtype, device=smem.device)
                    mem_embeds = torch.cat([mem_embeds, smem], dim=1)
                # Concat LTM if STM does not exist
                elif lmem is not None:
                    mem_embeds = mem_embeds.to(dtype=lmem.dtype, device=lmem.device)
                    mem_embeds = torch.cat([mem_embeds, lmem], dim=1)
            elif "ltm_over" in self.memory_zip_method.lower():
                # Concat LTM
                if lmem is not None:
                    mem_embeds = mem_embeds.to(dtype=lmem.dtype, device=lmem.device)
                    mem_embeds = torch.cat([mem_embeds, lmem], dim=1)
                # Concat STM if LTM does not exist
                elif smem is not None:
                    smem = smem.flatten(1, 2)
                    mem_embeds = mem_embeds.to(dtype=smem.dtype, device=smem.device)
                    mem_embeds = torch.cat([mem_embeds, smem], dim=1)
            else:
                raise ValueError(f"Memory zip method {self.memory_zip_method} does not exist")
        
        return mem_embeds
        

    def clear_states(self):
        printr(0, "Clearing memory")
        # demo 中取消该函数的所有内容
        self.stm.clear_memory()
        self.ltm.clear_memory()
        if self.training:
            self.token_cnt_before_drops.clear()
            self.token_cnt_after_drops.clear()


if __name__ == '__main__':
    vid1 = torch.randn(3, 8, 8, 2)
    vid2 = torch.randn(5, 9, 10, 2)
    vid3 = torch.randn(4, 10, 12, 2)
    vid4 = torch.randn(4, 11, 12, 2)
    vid5 = torch.randn(6, 8, 13, 2)

    stm = ShortTermMemory(storage=11, model_dim=2, output_channels=10)
    ltm = LongTermMemory(
        spatial_comp_xa_config=InterFrameDecoderConfig(num_queries=4, d_model=10, num_heads=2, ff_dim=24),
        inter_frame_config=InterFrameDecoderConfig(num_queries=9, d_model=10, num_heads=2, ff_dim=24),
        update_curr_config=InterClipDecoderConfig(d_model=10, num_heads=2, ff_dim=24)
    )

    mm = Memories(stm, ltm)

    print("--- offline mode ---")
    sm, lm = mm.process_videos_embeddings(None, [vid1, vid2, vid3, vid4, vid5])
    print(f"{sm.shape=}, {lm.shape=}")

    # stm = ShortTermMemory(frame_storage=2, num_windows_per_side=2, model_dim=2, output_channels=10)
    # ltm = LongTermMemory(
    #     sa_config=SelfAttnDecoderConfig(d_model=10, num_heads=2, ff_dim=24),
    #     xa_compress_config=CompressiveCrossAttnDecoderConfig(num_queries=9, d_model=10, num_heads=2, ff_dim=24),
    #     xa_historic_config=HistoricalCrossAttnDecoderConfig(d_model=10, num_heads=2, ff_dim=24)
    # )
    # mm = MemoryManager(stm, ltm, streaming_mode=True)

    # print("--- online mode ---")
    # mem = mm.process_videos_embeddings(vid1)
    # print(f"{mem.shape=}")
    # mem = mm.process_videos_embeddings(vid2)
    # print(f"{mem.shape=}")
    # mem = mm.process_videos_embeddings(vid3)
    # print(f"{mem.shape=}")
    # mem = mm.process_videos_embeddings(vid4)
    # print(f"{mem.shape=}")
    # mem = mm.process_videos_embeddings(vid5)
    # print(f"{mem.shape=}")