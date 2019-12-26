import torch
import torch.nn as nn
import numpy as np
from transformer import Transformer
from encoder_layer import EncoderLayer


def test_transformer_cell():
    # this is just a smoke test; these modules are implemented through
    # autograd so no Jacobian test is needed
    d_model = 512
    nhead = 16
    num_encoder_layers = 4
    num_decoder_layers = 3
    dim_feedforward = 256
    dropout = 0.3
    bsz = 8
    seq_length = 35
    tgt_length = 15

    transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                 dim_feedforward, dropout)
    src = torch.randn(seq_length, bsz, d_model)
    src_mask = transformer.generate_square_subsequent_mask(seq_length).double()
    tgt = torch.randn(tgt_length, bsz, d_model)
    tgt_mask = transformer.generate_square_subsequent_mask(tgt_length).double()
    memory_mask = torch.randn(tgt_length, seq_length).double()
    src_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5
    tgt_key_padding_mask = torch.rand(bsz, tgt_length) >= 0.5
    memory_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5

    output = transformer(src, tgt,
                         src_mask=src_mask,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
    output.sum().backward()


def test_transformerencoderlayer():
    # this is a deterministic test for TransformerEncoderLayer
    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2

    model = EncoderLayer(d_model, nhead, dim_feedforward, dropout)

    # set constant weights of the model
    for idx, p in enumerate(model.parameters()):
        x = p.data
        sz = x.view(-1).size(0)
        shape = x.shape
        x = torch.cos(torch.arange(0, sz).float().view(shape))
        p.data.copy_(x)

    # deterministic input
    encoder_input = torch.Tensor([[[20, 30, 40, 50]]])
    result = model(encoder_input)
    ref_output = torch.Tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]])
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)
    # 0 values are NOT masked. This shouldn't mask anything.
    mask = torch.Tensor([[0]]) == 1
    result = model(encoder_input, src_key_padding_mask=mask)
    result = result.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)
    # 1 values are masked. Since there is only 1 input embedding this
    # will result in nan.
    mask = torch.Tensor([[1]]) == 1
    result = model(encoder_input, src_key_padding_mask=mask)
    result = result.detach().numpy()
    # self.assertTrue(np.isnan(result).all())

    # deterministic input
    encoder_input = torch.Tensor([[[1, 2, 3, 4]],
                                  [[5, 6, 7, 8]]])
    result = model(encoder_input)
    ref_output = torch.Tensor([[[2.272644, 0.119035, -0.691669, 0.153486]],
                               [[2.272644, 0.119035, -0.691669, 0.153486]]])
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)
    # all 0 which is no masking
    mask = torch.Tensor([[0, 0]]) == 1
    result = model(encoder_input, src_key_padding_mask=mask)
    result = result.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)
    mask = torch.Tensor([[1, 0]]) == 1
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = torch.Tensor([[[2.301516, 0.092249, -0.679101, 0.103088]],
                               [[2.301516, 0.092249, -0.679101, 0.103088]]])
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)

    # deterministic input
    encoder_input = torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                   [0.3718, 0.4945, 0.9511, 0.0864]]])
    result = model(encoder_input)
    ref_output = torch.Tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                [2.427987, 0.021213, -0.602496, -0.084103]],
                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                [2.413863, 0.022211, -0.612486, -0.072490]],
                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                [2.425104, 0.019748, -0.604515, -0.084839]],
                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                [2.433556, 0.021891, -0.598509, -0.086832]],
                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                [2.422901, 0.024187, -0.606178, -0.074929]]])
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)
    # all 0
    mask = torch.zeros([2, 5]) == 1
    result = model(encoder_input, src_key_padding_mask=mask)
    result = result.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)
    mask[0, 1] = 1
    mask[1, 3] = 1
    mask[1, 4] = 1
    result = model(encoder_input, src_key_padding_mask=mask)
    ref_output = torch.Tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                [2.428811, 0.021445, -0.601912, -0.084252]],
                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                [2.415408, 0.02249 , -0.611415, -0.073]],
                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                [2.42598, 0.019941, -0.603896, -0.085091]],
                               [[2.436457, 0.022736, -0.59643 , -0.08736],
                                [2.434021, 0.022093, -0.598179, -0.08679]],
                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                [2.4242, 0.024653, -0.605266, -0.074959]]])
    result = result.detach().numpy()
    ref_output = ref_output.detach().numpy()
    # self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
    np.testing.assert_allclose(result, ref_output, atol=1e-5)


test_transformerencoderlayer()