import random
import numpy as np
import torch
from torch.nn import functional as F
import seaborn as sns
from torch.utils.data.dataloader import DataLoader
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        print("probs: ", probs)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


def trim_tokens(tokens, bos, eos, pad=None):
    bos_idx = np.where(tokens == bos)[0]
    tokens = tokens[bos_idx[0]+1:] if len(bos_idx) > 0 else tokens
    eos_idx = np.where(tokens == eos)[0]
    tokens = tokens[:eos_idx[0]] if len(eos_idx) > 0 else tokens
    # tokens = tokens[tokens != bos]
    # tokens = tokens[tokens != eos]
    if pad is not None:
        tokens = tokens[tokens != pad]
    return tokens

class EvalConfig:
    # optimization parameters
    max_epochs = 100
    batch_size = 64

    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 4  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Evaluate:
    def __init__(self, model, valid_dataset, evalConfig, args):
        self.model = model
        self.valid_dataset = valid_dataset
        self.evalConfig = evalConfig
        
        print("Using wandb")
        wandb.login(key="aabe3a9de8b348d83b37bd4d1cbbdcd366f55c9e")
        wandb.init(project='LayoutTransformer', name=args.exp)
        wandb.config.update(args)
    
    def eval(self):
        self.model.eval()

        # Create a tensor of shape [4, 1] with each element having the value of 'bos'
        init_condition = torch.full((4, 1), self.valid_dataset.bos_token)

        # Print the tensor
        print(init_condition)

        # samples - random
        layouts = sample(self.model, init_condition, steps=self.valid_dataset.max_length,
                                    temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        sample_random_layouts = [self.valid_dataset.render(layout) for layout in layouts]

        # samples - deterministic
        layouts = sample(self.model, init_condition, steps=self.valid_dataset.max_length,
                            temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
        sample_det_layouts = [self.valid_dataset.render(layout) for layout in layouts]

        wandb.log({
                        "sample_random_layouts": [wandb.Image(pil, caption=f'bos_sample_random_{i:02d}.png')
                                                for i, pil in enumerate(sample_random_layouts)],
                        "sample_det_layouts": [wandb.Image(pil, caption=f'bos_sample_det_{i:02d}.png')
                                            for i, pil in enumerate(sample_det_layouts)],
                    }, step=self.valid_dataset.max_length)