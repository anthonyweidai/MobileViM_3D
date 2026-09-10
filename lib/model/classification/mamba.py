# Copyright (c) 2024, Tri Dao, Albert Gu.
import math
import torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton import (
    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, RMSNormGated,
)


class BaseMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        bias=False,
        layer_idx=None,
        use_mem_eff_path=True,  # Fused kernel options
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.activation = "swish"
        self.act = nn.SiLU()

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None) -> Tensor:
        raise NotImplementedError
    

class Mamba2(BaseMamba):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=1,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        layer_idx=None,  # Absorb kwarg for general module
        use_mem_eff_path=True,
        DualDirection=False,
        IsCatProj = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model, d_state, d_conv, expand, bias, layer_idx,
            use_mem_eff_path, **factory_kwargs,
        )
        self.conv_init = conv_init
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.chunk_size = chunk_size
        A_init_range = (1, d_state)
        
        self.IsCatProj = IsCatProj
        self.DualDirection = DualDirection

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True
            
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
        
        if DualDirection:
            AReverse = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            AReverseLog = torch.log(AReverse).to(dtype=dtype)
            self.AReverseLog = nn.Parameter(AReverseLog)
            self.AReverseLog._no_weight_decay = True

            self.Conv1dReverse = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            
            self.DReverse = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.DReverse._no_weight_decay = True
            
            if self.IsCatProj:
                self.CatProj = nn.Linear(self.d_model * 2, self.d_model, bias=bias, **factory_kwargs)
            
        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

    def forward(self, u, seq_idx=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path:
            # Fully fused path
            Output = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            if self.DualDirection:
                AReverse = -torch.exp(self.AReverseLog)  # (nheads) or (d_inner, d_state)
                OutReverse = mamba_split_conv1d_scan_combined(
                    zxbcdt.flip([-1]),
                    rearrange(self.Conv1dReverse.weight, "d 1 w -> d w"),
                    self.Conv1dReverse.bias,
                    self.dt_bias,
                    AReverse,
                    D=self.DReverse,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
                if self.IsCatProj:
                    Output = self.CatProj(torch.cat([Output, OutReverse.flip([-1])], dim=-1))
                else:
                    Output = (Output + OutReverse.flip([-1])) / 2
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            Output = self.out_proj(y)
        return Output