import torch
from transformers import GPT2Config, GPT2Model # type: ignore
from torch import nn

from core import ContextModel

def relu_attn(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
  
    if self.scale_attn_weights:
      attn_weights = attn_weights / (value.size(-1) ** 0.5)
  
      # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
      attn_weights = attn_weights / float(self.layer_idx + 1)
  
    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
      query_length, key_length = query.size(-2), key.size(-2)
      causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
      attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
  
    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask
  
    attn_weights = nn.functional.relu(attn_weights) / query.size(-2) 
  
    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
  
    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
  
    attn_output = torch.matmul(attn_weights, value)
  
    return attn_output, attn_weights

def relu_attn_causal(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
  
    if self.scale_attn_weights:
        attn_weights = attn_weights / (value.size(-1) ** 0.5)
  
    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)
  
    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
  
    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask
  
      # TODO: make this sequence length causal (divide by tokens seen so far, not total tokens in sequence)
      # relud = nn.functional.relu(attn_weights)
    seq_len = query.size(-2)
    causal_seq_len = 1 + ( torch.arange(seq_len, device=DEVICE)
                                  .expand(attn_weights.shape)
                                  .transpose(-1, -2) )
      # import code
      # assert attn_weights.shape == causal_seq_len.shape, code.interact(local=locals(), banner=f"Failed shape check: attn_weights do not math causal_seq_len in shape! \n{attn_weights.shape} vs {causal_seq_len.shape}")
      # pre_attn_weights = attn_weights
    attn_weights = nn.functional.relu(attn_weights) / (causal_seq_len + 1)
      # code.interact(local=locals(), banner="yeesh")
  
      # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
  
      # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
  
    attn_output = torch.matmul(attn_weights, value)
  
    return attn_output, attn_weights
      
class TransformerModel(ContextModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            no_attention=True,
            want_pos_embeddings=False
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.context_length = n_positions
        self._n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration, attn_func=relu_attn)
        self._read_out = nn.Linear(n_embd, 1)

    def forward(self, xs, ys):
        inds = torch.arange(ys.shape[1])
        
        zs = ContextModel.interleave(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
