import torch
import torch.nn as nn
from model.FourierAttModules import Encoder, LayerNorm

class FourierAtt(nn.Module):
    def __init__(self, args):
        super(FourierAtt, self).__init__()
        self.args = args
        self.item_encoder = Encoder(args)

        self.apply(self.init_weights)

    # same as SASRec
    def forward(self, input_ids):
        item_encoded_layers = self.item_encoder(input_ids,
                                                # extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
