"""
=========================================================================================
 CNN + MultiConv_MLP architecture reference for TokaMark task_1-3
=========================================================================================

WHAT THIS FILE IS
-----------------
A *self-contained, readable* reconstruction of the exact model the baseline trains
(`MultiConv_MLP`) plus the per-channel CNN encoder/decoder blocks it is built from.
It mirrors the upstream source (tokamark_baseline@1e7c489):
    src/multi_conv_mlp_model.py
    src/conv_encoders_decoders.py
...but strips the framework glue so you can read *just the architecture* end to end.

It is NOT imported by the pipeline. It is a learning/reference artifact.
Verbatim upstream copies live next to this file in docs/reference/.

HOW TO READ THIS FILE
---------------------
1. The `task_1-3` shape contract (START HERE) -> what goes in, what comes out.
2. `Conv1DEncoder` -> how one signal channel is turned into a vector.
3. `MultiConv_MLP` -> how all channels are combined and decoded to psi.
4. The "worked shape trace" at the bottom -> concrete numbers for task_1-3.

NOTE ON DEVICE: upstream calls get_device() at import. Here we take `device` as an
argument / default to CPU so the file does not need a GPU to be read or reason about.

=========================================================================================
 0. THE task_1-3 SHAPE CONTRACT  (START HERE)
=========================================================================================

Task config (task_1-3.yaml) says, in order:
  input_length  = 0.005      (seconds of input window)
  output_length = 0.005      (seconds of output window)
  delta         = -0.005     (input window ends where output window begins)
  stride_window = 0.005      (window step)
  task_type     = markovian  (single window -> single window; no recurrence needed)

Inputs  : 8 signals, each shaped (C, T) where T = time samples in 0.005 s.
Output  : equilibrium/psi, a 65 x 65 spatial field for the same window.

Per-channel shapes seen at runtime (smoke test), as (C, T):
  [0] magnetics/flux_loop_flux                    (1, 25)
  [1] magnetics/b_field_pol_probe_ccbv_field      (1, 25)
  [2] magnetics/b_field_pol_probe_obr_field       (1, 25)
  [3] magnetics/b_field_pol_probe_obv_field       (1, 25)
  [4] magnetics/b_field_tor_probe_saddle_voltage  (1, 250)
  [5] pf_active/coil_current                      (1, 20)
  [6] pf_active/solenoid_current                  (1, 20)
  [7] summary/ip                                  (1, 20)

  x batch shape : list of 8 tensors, each (B, 1, 1, T)   [see ModelTransform_1]
  y batch shape : (B, 1, 65, 65)

So at the model boundary (after the transforms), every input is:
    (B, W, C, L)  with W = 1 window, C = 1 channel, L = T samples
i.e. a 1-D signal with a length channel. That is why Conv1D paths are used.

=========================================================================================
 1. BUILDING BLOCK: Conv1DEncoder
=========================================================================================

One such block per input channel. It compresses (C, T) -> a fixed vector of size D*bb.

Key hyper-parameters (module-level defaults):
    PADDING      = 1
    KERNEL_SIZE  = 3
    STRIDE       = 3
    LAYERS       = 3
    BB_FACTOR    = 2   (model file uses 2; conv file default is 3 -- model wins)

Internal op order, repeated `layers` times:
    Conv1d(C -> out, k=3, s=3, p=1) -> ReLU -> MaxPool1d(2, p=1) -> BatchNorm1d
with channels: in=C; first conv emits D, then D*2, D*4 ... (out *= 2 each layer).
After the loop, flatten and a single Linear maps to D*bb. That is the "bottleneck vector".
"""

import torch
import torch.nn as nn

PADDING = 1
KERNEL_SIZE = 3
STRIDE = 3
LAYERS_ENCODER = 3
LAYERS_DECODER = 3
BB_FACTOR = 2


def compressed_size_encoder(L, layers, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING):
    """Length of a 1-D axis after `layers` x [Conv1d then MaxPool1d(2,p=1)]. Used to size the FC."""
    for _ in range(layers):
        L = (L + 2 * padding - kernel_size) // stride + 1      # Conv1d output
        L = (L + 2 * padding - 2) // 2 + 1                      # MaxPool1d(2, p=1)
    return L


def list_compressed_size_decoder(L_out, layers, kernel_size=KERNEL_SIZE, stride=STRIDE,
                                 padding=PADDING, output_padding=1):
    """The sequence of axis lengths the decoder must hit after each ConvTranspose1d.
    Returned reversed so index 0 is the length the FC must produce (deepest/smallest)."""
    import math
    L = L_out
    seq = [L]
    for _ in range(layers):
        L = (L + 2 * padding - kernel_size + output_padding) // stride + 1
        L = int(math.ceil(L))
        seq.append(L)
    return seq[::-1]


class Conv1DEncoder(nn.Module):
    """(C, T) signal -> vector of size D*bb_factor.

    Example (task_1-3 channel 0): C=1, T=25, D=16, layers=3, bb=2.
      ts_comp = compressed_size_encoder(25, 3) = 1
      channels: 1 -> 16 -> 32 -> 64 (final = D * 2**(layers-1) = 64)
      fc: (64 * 1) -> 16*2 = 32
    """

    def __init__(self, input_shape, D, layers=LAYERS_ENCODER, kernel_size=KERNEL_SIZE,
                 stride=STRIDE, padding=PADDING, bb_factor=BB_FACTOR):
        super().__init__()
        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]     # channels, e.g. 1
        self.ts_var = input_shape[1]    # time samples, e.g. 25

        self.ts_comp = compressed_size_encoder(self.ts_var, layers, kernel_size, stride, padding)

        modules = [nn.BatchNorm1d(self.n_var)]      # normalise the raw signal first
        in_channels = self.n_var
        out_channels = D
        for _ in range(layers):
            modules.append(nn.Conv1d(in_channels, out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool1d(2, padding=padding))
            modules.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
            out_channels *= 2
        self.cnn = nn.Sequential(*modules)

        final_channels = D * (2 ** (layers - 1))
        self.fc = nn.Linear(final_channels * self.ts_comp, D * bb_factor)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class Conv1DDecoder(nn.Module):
    """Vector of size D*bb_factor -> (n_var, ts_var) signal.

    Mirror image of the encoder, using ConvTranspose1d. The FC first expands the
    latent vector back to (final_channels, smallest_len), then each transpose-conv
    upsamples and is CROPPED to the exact target length (so we always land on the
    requested output size, e.g. 65).
    """

    def __init__(self, output_shape, D, layers=LAYERS_DECODER, kernel_size=KERNEL_SIZE,
                 stride=STRIDE, padding=PADDING, bb_factor=BB_FACTOR, output_padding=1):
        super().__init__()
        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]

        self.list_ts_comp = list_compressed_size_decoder(
            self.ts_var, layers, kernel_size, stride, padding, output_padding)

        final_channels = D * (2 ** (layers - 1))
        self.fc = nn.Linear(D * bb_factor, final_channels * self.list_ts_comp[0])

        modules = []
        in_channels = final_channels
        for i in range(layers):
            out_channels = in_channels // 2 if i < layers - 1 else self.n_var
            modules.append(nn.ConvTranspose1d(in_channels, out_channels,
                                              kernel_size=kernel_size, stride=stride,
                                              padding=padding, output_padding=output_padding))
            if i < layers - 1:
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        self.transposecnn = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.D * (2 ** (self.layers - 1)), self.list_ts_comp[0])
        conv_id = 0
        for layer in self.transposecnn:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose1d):
                conv_id += 1
                x = x[:, :, :self.list_ts_comp[conv_id]]   # crop to exact target length
        return x


# NOTE: upstream also has Conv2DEncoder/2DDecoder and Conv3DEncoder/3DDecoder (see
# docs/reference/conv_encoders_decoders_upstream.py). They are structurally identical
# but operate on (C, T, H) and (C, T, H, W) respectively -- used by other TokaMark
# tasks that take profiles/images. task_1-3 uses the 1-D variants only.


# =========================================================================================
# 2. THE MODEL: MultiConv_MLP
# =========================================================================================
#
# High level:
#   (a) each input channel -> its own Conv1DEncoder -> vector of size D*bb
#   (b) all channel vectors concatenated along the feature dim -> (B, W, n_channels*D*bb)
#   (c) encoder MLP collapses this to a single context vector (B, D*bb)
#   (d) decoder MLP turns context (+ optional exogenous) into a latent (B, D*bb)
#   (e) latent is repeated across output time steps -> (B, W_out, D*bb)
#   (f) each output channel has a Conv1DDecoder -> predicted field (B, W_out, C_out, L_out)
#
# task_1-3 specifics:
#   n_channels   = 8 inputs, 0 exogenous (the "exogenous" list is empty here)
#   y shape      = (1, 65, 65) -> after _make_dummy_outputs becomes a 1-D axis of 65
#   so W_out=1, and the single output decoder produces psi of length 65 x ... see trace.


class MultiConv_MLP(nn.Module):
    def __init__(self, input_shapes, exogenous_shapes, output_shapes,
                 output_latent_shapes, W_in, W_out, D=16):
        """
        input_shapes / exogenous_shapes / output_shapes : the runtime shapes discovered
            by create_cnn_architecture() when it peeks the first sample.
        output_latent_shapes : shapes AFTER _make_dummy_outputs (output reshaped for decoding).
        W_in / W_out : number of input windows / output windows (1 and 1 for task_1-3).
        D : base channel width (16).
        """
        super().__init__()
        self.D = D
        self.W_in = W_in
        self.W_out = W_out

        # ---- per-input-channel encoders ----------------------------------------------
        self.input_branches = nn.ModuleList()
        for var_shape in input_shapes:
            # var_shape here is (C, T); a 3-tuple triggers the 1-D encoder in upstream.
            self.input_branches.append(
                Conv1DEncoder(var_shape, D, LAYERS_ENCODER, KERNEL_SIZE, STRIDE, PADDING, BB_FACTOR)
            )

        # ---- the MLP "bottleneck" -----------------------------------------------------
        # encoder: (n_in * D * bb * W_in) -> 2*D*bb -> D*bb
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D * BB_FACTOR * len(input_shapes) * W_in, 2 * D * BB_FACTOR),
            nn.ReLU(),
            nn.Linear(2 * D * BB_FACTOR, D * BB_FACTOR),
        )
        # decoder: (D*bb * (1 + n_exo*W_out)) -> 2*D*bb -> D*bb
        self.decoder_mlp = nn.Sequential(
            nn.Linear(D * BB_FACTOR * (1 + len(exogenous_shapes) * W_out), 2 * D * BB_FACTOR),
            nn.ReLU(),
            nn.Linear(2 * D * BB_FACTOR, D * BB_FACTOR),
        )

        # ---- exogenous encoders (empty for task_1-3) ---------------------------------
        self.exogenous_branches = nn.ModuleList()
        for var_shape in exogenous_shapes:
            self.exogenous_branches.append(
                Conv1DEncoder(var_shape, D, LAYERS_ENCODER, KERNEL_SIZE, STRIDE, PADDING, BB_FACTOR)
            )

        # ---- per-output-channel decoders ---------------------------------------------
        self.output_branches = nn.ModuleList()
        for var_shape in output_latent_shapes:
            self.output_branches.append(
                Conv1DDecoder(var_shape, D, LAYERS_DECODER, KERNEL_SIZE, STRIDE, PADDING, BB_FACTOR)
            )

        self.output_shapes = output_shapes

    @staticmethod
    def _run_cnn_encoder(branch, x):
        """x: (B, W, C, L). Fold W into the batch for the CNN, then unfold back."""
        B, W = x.shape[:2]
        x = x.view(B * W, *x.shape[2:])
        out = branch(x)
        return out.view(B, W, -1)

    @staticmethod
    def _run_cnn_decoder(branch, goal_shape, x):
        """x: (B, W, D*bb). Decode, then keep the last `goal_shape` time steps."""
        B, W = x.shape[:2]
        x = x.reshape(B * W, *x.shape[2:])
        out = branch(x)
        out = out.reshape(B, W * out.shape[2], *out.shape[3:])
        target_len = goal_shape[0]
        return out[:, -target_len:]

    def forward(self, *args):
        n_in = len(self.input_branches)
        n_exo = len(self.exogenous_branches)
        inputs = args[:n_in]
        exogenous = args[n_in:n_in + n_exo]

        # (a) encode each input channel -> (B, W, D*bb); concat along feature dim
        enc = [self._run_cnn_encoder(b, x) for b, x in zip(self.input_branches, inputs)]
        input_seq = torch.cat(enc, dim=2)                      # (B, W, n_in*D*bb)

        # exogenous (empty for task_1-3)
        exo_seq = None
        if n_exo > 0:
            exo = [self._run_cnn_encoder(b, x) for b, x in zip(self.exogenous_branches, exogenous)]
            exo_seq = torch.cat(exo, dim=2)

        B, W_in, D_in = input_seq.shape

        # (b)+(c) flatten windows+features, MLP down to context (B, D*bb)
        enc_in = input_seq.reshape(B, W_in * D_in)
        context = self.encoder_mlp(enc_in)

        # (d) decoder input = context (+ exogenous flattened)
        if exo_seq is not None:
            B, W_out, D_exo = exo_seq.shape
            decoder_in = torch.cat([context, exo_seq.reshape(B, W_out * D_exo)], dim=1)
        else:
            decoder_in = context
        dec_flat = self.decoder_mlp(decoder_in)                 # (B, D*bb)

        # (e) broadcast the latent across output windows
        dec_out = dec_flat.unsqueeze(1).repeat(1, self.W_out, 1)   # (B, W_out, D*bb)

        # (f) decode each output channel
        outputs = []
        for branch, goal in zip(self.output_branches, self.output_shapes):
            outputs.append(self._run_cnn_decoder(branch, goal, dec_out))
        return outputs


# =========================================================================================
# 3. WORKED SHAPE TRACE for task_1-3  (D=16, bb=2, layers=3)
# =========================================================================================
#
# Input branch (per channel), e.g. channel 0 (1, 25):
#   BN(1) -> Conv1d(1->16,k3,s3,p1) -> ReLU -> MaxPool(2,p1) -> BN(16)      # len 25 -> 9 -> 5
#          -> Conv1d(16->32,...)   -> ReLU -> MaxPool -> BN(32)             # 5 -> 2 -> 2
#          -> Conv1d(32->64,...)   -> ReLU -> MaxPool -> BN(64)             # 2 -> 1 -> 1
#   compressed_size_encoder(25,3) = 1 ; final_channels = 64
#   flatten -> 64*1 ; fc -> 32 (=D*bb).  So each channel yields a 32-vector.
#
# 8 input channels -> concat -> 8*32 = 256 features per window.
#   encoder_mlp: 256 -> 64 -> 32      (W_in = 1)
#   decoder_mlp: 32  -> 64 -> 32      (no exogenous)
#   broadcast to W_out = 1 window -> (B, 1, 32)
#
# Output decoder for psi:
#   _make_dummy_outputs reshapes (1,65,65) into a decodable latent. The decoder's
#   Conv1DDecoder is sized from output_latent_shapes, whose length axis is 65 here.
#   list_compressed_size_decoder(65, 3, out_pad=1) -> [3, 8, 22, 65] (deepest -> final).
#   fc: 32 -> final_channels(64) * 3 ; then ConvTranspose1d x3, each cropped
#   back to the exact ladder length (3 -> 8 -> 22 -> 65).
#
# Final output: (B, 1, 65, 65)  -> the reconstructed psi field.
#
# Parameter scale: this whole model is ~0.3 M params (measured: "0.3M params" at init),
# which is tiny by deep-learning standards -- one reason it can memorise a small subset.
#
# =========================================================================================
