import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    projection 레이어를 처음 forward() 호출될 때 자동으로 만듭니다.
    """
    def __init__(self, configs):
        super().__init__()

        # 기본 설정
        self.task_name        = configs.task_name
        self.pred_len         = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # Encoder (FFT-based 인코더가 아니라 일반 Transformer)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # projection은 lazy init
        self.projection = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Encoder 출력을 projection 레이어로 변환.
        projection 레이어는 out_dim = x_dec.size(-1) 에 맞춰
        최초 forecast 호출 시에만 생성됩니다.
        """
        # 1) encoder
        enc_in = self.enc_embedding(x_enc, x_mark_enc)  # [B, L_enc, d_model]
        enc_out, _ = self.encoder(enc_in, attn_mask=None)  # [B, L_enc, d_model]

        # 2) lazy projection init
        out_dim = x_dec.size(-1)  # 예측하려는 채널 수
        if self.projection is None:
            self.projection = nn.Linear(
                enc_out.size(-1),  # d_model
                out_dim,
                bias=True
            ).to(enc_out.device)

        # 3) decode
        dec_out = self.projection(enc_out)  # [B, L_enc, out_dim]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        전체 포워드. decoder 입력 x_dec 은 실제 projection에는 사용되지 않습니다.
        마지막 pred_len 시점만 slice 해서 반환합니다.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # 뒤쪽 pred_len 타임스텝만 리턴
        return dec_out[:, -self.pred_len:, :]
