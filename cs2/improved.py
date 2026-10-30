"""Final region-aligned readout model and its conditional count loss."""
import math


import torch


from torch import nn


from torch.nn import functional as F


def conditional_ranked_score(logits, target):
    """RPS over ordered counts 0..5; fixed normalization by five thresholds."""
    cdf = logits.float().softmax(-1).cumsum(-1)[...,:5]
    observed = (torch.arange(5,device=target.device) >= target[...,None]).to(cdf.dtype)
    return (cdf-observed).square().mean(-1)


class RegionalReadout(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.heads,self.width = heads,dim//heads
        self.norm = nn.LayerNorm(dim)
        self.query,self.key,self.value = [nn.Linear(dim,dim) for _ in range(3)]
        self.output = nn.Linear(dim,dim)

    def forward(self,query,memory,local):
        b,k,dim = memory.shape
        keys = self.norm(memory)
        q = self.query(self.norm(query)).reshape(b,self.heads,self.width)
        kk = self.key(keys).reshape(b,k,self.heads,self.width).transpose(1,2)
        vv = self.value(keys).reshape(b,k,self.heads,self.width).transpose(1,2)
        weights = ((q.float()[:,:,None,:]*kk.float()).sum(-1)/math.sqrt(self.width)).softmax(-1)
        attended = (weights.to(vv.dtype)[...,None]*vv).sum(-2).reshape(b,dim)
        return self.output(attended+self.norm(local)),weights


class ImprovedTransformer(nn.Module):
    """Keep region tokens through count allocation instead of only global pooling."""
    exact_nll = True
    supports_amp = True
    def __init__(self,features,k,cfg):
        # The original encoder/decoder parameters have matched seeded starts.
        super().__init__()
        from .models import TacticalTransformer
        original = TacticalTransformer(features,k,cfg)
        for name,module in original.named_children(): self.add_module(name,module)
        self.k = k
        dim = cfg.get("transformer_dim",cfg["hidden_dim"])
        self.regional_readout = RegionalReadout(dim,cfg.get("transformer_heads",4))
        self.readout_scale = nn.Parameter(torch.full((dim,),.05))
        self.rps_weight = float(cfg.get("conditional_rps_weight",.25))
        if self.rps_weight < 0: raise ValueError("RPS weight must be nonnegative")
        self.readout_enabled = True
        self.capture_readout = False

    def encode_context(self,x):
        return self.encoder(x,return_regions=True)

    def allocation_logits(self,state,memory,remaining,region):
        """Conditional count logits and optional mechanism diagnostics."""
        return self.head(state),{}

    def begin_allocation(self,memory,n):
        return None

    def conditional_allocation(self,state,memory,remaining,region,context):
        return self.allocation_logits(state,memory,remaining,region)

    def allocation_memory(self,memory,remaining,region,context):
        return memory

    def observe_allocation(self,context,value,logits,region):
        return context

    def decode(self,ctx,n,y=None,return_aux=False):
        pooled,memory = ctx
        state,remaining,previous = pooled,n.clone(),torch.zeros_like(n)
        output,nll,rps = [],torch.zeros_like(n,dtype=torch.float32),torch.zeros_like(n,dtype=torch.float32)
        options = torch.arange(6,device=n.device)[None]
        traces = []
        active = self.readout_enabled and not getattr(self,"diagnostic_disable_readout",False)
        allocation_context = self.begin_allocation(memory,n)
        for region in range(self.k-1):
            step_memory = self.allocation_memory(memory,remaining,region,allocation_context)
            ids = torch.full_like(n,region)
            embedded = self.region(ids)+self.count(previous)+self.remaining(remaining)
            if active:
                update,weights = self.regional_readout(state+embedded+step_memory[:,region],step_memory,step_memory[:,region])
                residual = self.readout_scale.tanh()*update
                embedded = embedded+residual
            else:
                weights = memory.new_zeros(len(n),self.regional_readout.heads,self.k)
                residual = torch.zeros_like(state)
            state = self.cell(embedded,state)
            logits,allocation_trace = self.conditional_allocation(state,step_memory,remaining,region,allocation_context)
            logits = logits.masked_fill(options > remaining[:,None],-1e9)
            value = y[:,region] if y is not None else torch.multinomial(logits.float().softmax(-1),1).squeeze(-1)
            nll = nll+F.cross_entropy(logits.float(),value,reduction="none")
            if y is not None: rps = rps+conditional_ranked_score(logits,value)
            if self.capture_readout:
                traces.append({"attention":weights.detach(),"residual_rms":residual.detach().float().square().mean(-1).sqrt(),
                               "remaining":remaining.detach().clone(),"probabilities":logits.detach().float().softmax(-1),
                               **{key:value.detach() for key,value in allocation_trace.items()}})
            output.append(value)
            allocation_context = self.observe_allocation(allocation_context,value,logits,region)
            previous,remaining = value,remaining-value
        output.append(remaining)
        if self.capture_readout:
            self.readout_trace = {key:torch.stack([r[key] for r in traces],1) for key in traces[0]}
        result = torch.stack(output,-1),nll
        return (*result,rps) if return_aux else result

    def nll(self,x,y,n):
        return self.decode(self.encode_context(x),n,y)[1]

    def loss(self,x,y,n):
        _,nll,rps = self.decode(self.encode_context(x),n,y,return_aux=True)
        return (nll+self.rps_weight*rps).mean()

    def sample_context(self,ctx,n,samples):
        expanded = tuple(v.repeat_interleave(samples,0) for v in ctx)
        return self.decode(expanded,n.repeat_interleave(samples))[0].reshape(len(n),samples,self.k).float()

    def sample(self,x,n,samples):
        return self.sample_context(self.encode_context(x),n,samples)

    def infer(self,x,y,n,samples):
        ctx = self.encode_context(x)
        return self.sample_context(ctx,n,samples),self.decode(ctx,n,y)[1]

