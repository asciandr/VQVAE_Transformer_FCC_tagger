import torch.nn as nn
N_FEAT=35

#Model definition
class JetTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=128, nhead=4, num_layers=4, num_classes=7):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, d_model)
        # test continuous PF features
        # as performance ceriling of pipeline
        self.input_proj = nn.Linear(N_FEAT, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            # add dropout 
            # to prevent overfitting
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, tokens, mask):
        # tokens: [B, N]
        x = self.embedding(tokens)   # [B, N, d_model]

        # attention mask (True = ignore)
        attn_mask = ~mask.bool()

        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # masked mean pooling
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        logits = self.classifier(x)
        return logits


