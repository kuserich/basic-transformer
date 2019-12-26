import copy
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        r"""TransformerEncoder is a stack of N encoder layers
        Args:
            encoder_layer: an instance of the TransformerEncoderLayer() class (required).
            num_layers: the number of sub-encoder-layers in the encoder (required).
            norm: the layer normalization component (optional).
        Examples::
            >>> encoder_layer = .EncoderLayer(d_model=512, nhead=8)
            >>> transformer_encoder = .Encoder(encoder_layer, num_layers=6)
            >>> src = torch.rand(10, 32, 512)
            >>> out = transformer_encoder(src)
        """
        __constants__ = ['norm']

        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])