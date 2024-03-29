import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
import math
import numpy as np
from inspect import isfunction
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from PIL import Image
#from einops import rearrange, reduce
from tqdm.auto import tqdm
from utils.ema_jittor import EMA
from utils.generate_result import num_to_groups

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# assign and copy require_grad attribute
def assign_keep_grad(dest, src):
    if dest.requires_grad:
        dest.assign(src).start_grad()
    else:
        dest.assign(src).stop_grad()

class SiLU(jt.Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return x.sigmoid()*x

def exists(x):
    return (x is not None)

def default(val, d):
    if exists(val):
        return val
    return (d() if isfunction(d) else d)

def cycle(dl):
    while True:
        for data in dl:
            (yield data)

def has_int_squareroot(num):
    return ((math.sqrt(num) ** 2) == num)

def convert_image_to(img_type, image):
    if (image.mode != img_type):
        return image.convert(img_type)
    return image

def l2norm(t):
    return jt.normalize(t, dim=(- 1))

def normalize_to_neg_one_to_one(img):
    return ((img * 2) - 1)

def unnormalize_to_zero_to_one(t):
    return ((t + 1) * 0.5)

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def execute(self, x, *args, **kwargs):
        return (self.fn(x, *args, **kwargs) + x)

def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv(dim, default(dim_out, dim), 3, padding=1))

def Downsample(dim, dim_out=None):
    return nn.Conv(dim, default(dim_out, dim), 4, stride=2, padding=1)

class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = jt.ones((1, dim, 1, 1))

    def execute(self, x):
        eps = (1e-05 if (x.dtype == jt.float32) else 0.001)
        var = jt.var(x, dim=1, unbiased=False, keepdims=True)
        mean = jt.mean(x, dim=1, keepdims=True)
        return (((x - mean) / (var + eps).sqrt()) * self.g)

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def execute(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def execute(self, x):
        half_dim = (self.dim // 2)
        emb = (math.log(10000) / (half_dim - 1))
        emb = jt.exp((jt.arange(half_dim) * (- emb)))
        emb = (x[:, None] * emb[None, :])
        emb = jt.contrib.concat((emb.sin(), emb.cos()), dim=(- 1))
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    " following @crowsonkb 's lead with learned sinusoidal pos emb "
    ' https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 '

    def __init__(self, dim):
        super().__init__()
        assert ((dim % 2) == 0)
        half_dim = (dim // 2)
        self.weights = jt.randn(half_dim)

    def execute(self, x):
        x = jt.unsqueeze(x, 1)
        freqs = (((x * jt.unsqeeze(self.weights, 0)) * 2) * math.pi)
        fouriered = jt.contrib.concat((freqs.sin(), freqs.cos()), dim=(- 1))
        fouriered = jt.contrib.concat((x, fouriered), dim=(- 1))
        return fouriered

class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = SiLU()

    def execute(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            (scale, shift) = scale_shift
            x = ((x * (scale + 1)) + shift)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (nn.Sequential(SiLU(), nn.Linear(time_emb_dim, (dim_out * 2))) if exists(time_emb_dim) else None)
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (nn.Conv(dim, dim_out, 1) if (dim != dim_out) else nn.Identity())

    def execute(self, x, time_emb=None):
        scale_shift = None
        if (exists(self.mlp) and exists(time_emb)):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(*time_emb.shape[0:2], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return (h + self.res_conv(x))

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = (dim_head ** (- 0.5))
        self.heads = heads
        hidden_dim = (dim_head * heads)
        self.to_qkv = nn.Conv(dim, (hidden_dim * 3), 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv(hidden_dim, dim, 1), LayerNorm(dim))

    def execute(self, x):
        (b, c, h, w) = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # (q, k, v) = map((lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads)), qkv)
        qkv = map((lambda t: t.view(t.shape[0], self.heads, -1, *t.shape[2:4])), qkv)
        (q, k, v) = map((lambda t: t.view(*t.shape[0:3], -1)), qkv)
        q = q.softmax(dim=(- 2))
        k = k.softmax(dim=(- 1))
        q = (q * self.scale)
        v = (v / (h * w))
        context = jt.linalg.einsum('bhdn,bhen->bhde', k, v)
        out = jt.linalg.einsum('bhde,bhdn->bhen', context, q)
        # out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = out.view(out.shape[0], -1, out.shape[3])
        out = out.view(*out.shape[0:2], h, w)
        return self.to_out(out)

class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, scale=16):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = (dim_head * heads)
        self.to_qkv = nn.Conv(dim, (hidden_dim * 3), 1, bias=False)
        self.to_out = nn.Conv(hidden_dim, dim, 1)

    def execute(self, x):
        (b, c, h, w) = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # (q, k, v) = map((lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads)), qkv)
        qkv = map((lambda t: t.view(t.shape[0], self.heads, -1, *t.shape[2:4])), qkv)
        (q, k, v) = map((lambda t: t.view(*t.shape[0:3], -1)), qkv)
        (q, k) = map(l2norm, (q, k))
        sim = jt.linalg.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        attn = sim.softmax(dim=(- 1))
        out = jt.linalg.einsum('bhij,bhdj->bhid', attn, v)
        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = out.permute((0, 1, 3, 2))
        out = out.view(out.shape[0], -1, out.shape[3])
        out = out.view(*out.shape[0:2], h, w)
        return self.to_out(out)

class Unet(nn.Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, resnet_block_groups=8, learned_variance=False, learned_sinusoidal_cond=False, learned_sinusoidal_dim=16):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map((lambda m: (dim * m)), dim_mults)]
        in_out = list(zip(dims[:(- 1)], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = (dim * 4)
        self.learned_sinusoidal_cond = learned_sinusoidal_cond
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = (learned_sinusoidal_dim + 1)
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for (ind, (dim_in, dim_out)) in enumerate(in_out):
            is_last = (ind >= (num_resolutions - 1))
            self.downs.append(nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim), block_klass(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), (Downsample(dim_in, dim_out) if (not is_last) else nn.Conv(dim_in, dim_out, 3, padding=1))]))
        mid_dim = dims[(- 1)]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for (ind, (dim_in, dim_out)) in enumerate(reversed(in_out)):
            is_last = (ind == (len(in_out) - 1))
            self.ups.append(nn.ModuleList([block_klass((dim_out + dim_in), dim_out, time_emb_dim=time_dim), block_klass((dim_out + dim_in), dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), (Upsample(dim_out, dim_in) if (not is_last) else nn.Conv(dim_out, dim_in, 3, padding=1))]))
        default_out_dim = (channels * (1 if (not learned_variance) else 2))
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass((dim * 2), dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv(dim, self.out_dim, 1)

    def execute(self, x, time):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for (block1, block2, attn, downsample) in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for (block1, block2, attn, upsample) in self.ups:
            x = jt.contrib.concat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = jt.contrib.concat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = jt.contrib.concat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

def extract(a, t, x_shape):
    (b, *_) = t.shape
    out = a.gather((- 1), t.float32())
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = (1000 / timesteps)
    beta_start = (scale * 0.0001)
    beta_end = (scale * 0.02)
    return jt.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    '\n    cosine schedule\n    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ\n    '
    steps = (timesteps + 1)
    x = jt.linspace(0, timesteps, steps)
    alphas_cumprod = (jt.cos((((((x / timesteps) + s) / (1 + s)) * math.pi) * 0.5)) ** 2)
    alphas_cumprod = (alphas_cumprod / alphas_cumprod[0])
    betas = (1 - (alphas_cumprod[1:] / alphas_cumprod[:(- 1)]))
    return jt.clamp(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):

    def __init__(self, model, *, image_size, channels=3, timesteps=1000, sampling_timesteps=None, loss_type='l1', objective='pred_noise', beta_schedule='cosine', p2_loss_weight_gamma=0.0, p2_loss_weight_k=1, ddim_sampling_eta=1.0):
        super().__init__()
        assert (not ((type(self) == GaussianDiffusion) and (model.channels != model.out_dim)))
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.objective = objective
        assert (objective in {'pred_noise', 'pred_x0'}), 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'
        if (beta_schedule == 'linear'):
            betas = linear_beta_schedule(timesteps)
        elif (beta_schedule == 'cosine'):
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        alphas = (1.0 - betas)
        alphas_cumprod = jt.cumprod(alphas, dim=0)
        alphas_cumprod_prev = nn.pad(alphas_cumprod[:(- 1)], (1, 0), value=1.0)
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert (self.sampling_timesteps <= timesteps)
        self.is_ddim_sampling = (self.sampling_timesteps < timesteps)
        self.ddim_sampling_eta = ddim_sampling_eta
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', jt.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', jt.sqrt((1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', jt.log((1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', jt.sqrt((1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', jt.sqrt(((1.0 / alphas_cumprod) - 1)))
        posterior_variance = ((betas * (1.0 - alphas_cumprod_prev)) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', jt.log(posterior_variance.clamp(min_v=1e-20)))
        self.register_buffer('posterior_mean_coef1', ((betas * jt.sqrt(alphas_cumprod_prev)) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', (((1.0 - alphas_cumprod_prev) * jt.sqrt(alphas)) / (1.0 - alphas_cumprod)))
        self.register_buffer('p2_loss_weight', ((p2_loss_weight_k + (alphas_cumprod / (1 - alphas_cumprod))) ** (- p2_loss_weight_gamma)))

    def predict_start_from_noise(self, x_t, t, noise):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) - (extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise))

    def predict_noise_from_start(self, x_t, t, x0):
        return (((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = ((extract(self.posterior_mean_coef1, t, x_t.shape) * x_start) + (extract(self.posterior_mean_coef2, t, x_t.shape) * x_t))
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def register_buffer(self, name, val):
        setattr(self, name, val.float32().stop_grad())

    def model_predictions(self, x, t):
        t = t.float32()
        model_output = self.model(x, t)
        if (self.objective == 'pred_noise'):
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
        elif (self.objective == 'pred_x0'):
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, clip_denoised: bool):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start
        if clip_denoised:
            #x_start.assign(x_start.clamp((- 1.0), 1.0))
            assign_keep_grad(x_start, x_start.clamp((- 1.0), 1.0))
        (model_mean, posterior_variance, posterior_log_variance) = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return (model_mean, posterior_variance, posterior_log_variance)

    @jt.no_grad()
    def p_sample(self, x, t: int, clip_denoised=True):
        (b, *_) = x.shape
        batched_times = jt.full((x.shape[0],), t, dtype='int64')
        (model_mean, _, model_log_variance) = self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = (jt.randn_like(x) if (t > 0) else 0.0)
        return (model_mean + ((0.5 * model_log_variance).exp() * noise))

    @jt.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]
        img = jt.randn(shape)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step'):
            img = self.p_sample(img, t)
        img = unnormalize_to_zero_to_one(img)
        return img

    @jt.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        (batch, total_timesteps, sampling_timesteps, eta, objective) = (shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective)
        times = jt.linspace(0.0, total_timesteps, steps=(sampling_timesteps + 2))[:(- 1)]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:(- 1)], times[1:]))
        img = jt.randn(shape)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]
            time_cond = jt.full((batch,), time, dtype='int64')
            (pred_noise, x_start, *_) = self.model_predictions(img, time_cond)
            if clip_denoised:
                #x_start.assign(x_start.clamp((- 1.0), 1.0))
                assign_keep_grad(x_start, x_start.clamp((- 1.0), 1.0))
            sigma = (eta * (((1 - (alpha / alpha_next)) * (1 - alpha_next)) / (1 - alpha)).sqrt())
            c = ((1 - alpha_next) - (sigma ** 2)).sqrt()
            noise = (jt.randn_like(img) if (time_next > 0) else 0.0)
            img = (((x_start * alpha_next.sqrt()) + (c * pred_noise)) + (sigma * noise))
        img = unnormalize_to_zero_to_one(img)
        return img

    @jt.no_grad()
    def sample(self, batch_size=16):
        (image_size, channels) = (self.image_size, self.channels)
        sample_fn = (self.p_sample_loop if (not self.is_ddim_sampling) else self.ddim_sample)
        return sample_fn((batch_size, channels, image_size, image_size))

    @jt.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        (b, *_) = x1.shape
        t = default(t, (self.num_timesteps - 1))
        assert (x1.shape == x2.shape)
        t_batched = jt.stack(([jt.Var(t)] * b))
        (xt1, xt2) = map((lambda x: self.q_sample(x, t=t_batched)), (x1, x2))
        img = (((1 - lam) * xt1) + (lam * xt2))
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, jt.full((b,), i, dtype='int64'))
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, (lambda : jt.randn_like(x_start)))
        return ((extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start) + (extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise))

    @property
    def loss_fn(self):
        if (self.loss_type == 'l1'):
            return lambda out, target, reduction: nn.l1_loss(out, target)
        elif (self.loss_type == 'l2'):
            return nn.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        (b, c, h, w) = x_start.shape
        noise = default(noise, (lambda : jt.randn_like(x_start)))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t)
        if (self.objective == 'pred_noise'):
            target = noise
        elif (self.objective == 'pred_x0'):
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction='none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.view(loss.shape[0], -1)
        loss = (loss * extract(self.p2_loss_weight, t, loss.shape))
        return loss.mean()

    def execute(self, img, *args, **kwargs):
        (b, c, h, w, img_size) = (*img.shape, self.image_size)
        assert ((h == img_size) and (w == img_size)), f'height and width of image must be {img_size}'
        t = jt.randint(0, self.num_timesteps, (b,)).int64()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t.float32(), *args, **kwargs)

class Trainer(object):

    def __init__(self, diffusion_model, dataset, *, gradient_accumulate_every=1, train_lr=0.0001, train_num_steps=100000, ema_update_every=10, ema_decay=0.995, adam_betas=(0.9, 0.99), save_and_sample_every=1000, num_samples=25, results_folder='./results', split_batches=True):

        super().__init__()
        self.model = diffusion_model
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.batch_size = dataset.batch_size
        self.ds = cycle(dataset)
        # only params in GaussianDiffusion.model is trainable
        # the custom register_buffer() should already do this, but enforce it
        # here again to be extra safe
        # trainable_params = set( param[1] for param in self.model.named_parameters() if 'model' in param[] )
        trainable_params = self.model.parameters()
        self.opt = jt.optim.Adam(trainable_params, lr=train_lr, betas=adam_betas)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.step = 0
        self.loss = []

    def save(self, milestone, model_ext='pkl'):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'loss': self.loss
        }
        jt.save(data, str((self.results_folder / f'model-{milestone}.{model_ext}')))

    def load(self, milestone, folder=None, model_ext='pkl'):
        if folder is None:
            folder = self.results_folder
        else:
            folder = Path(folder)
        data = jt.load(str((folder / f'model-{milestone}.{model_ext}')))
        self.step = data['step']
        self.model.load_parameters(data['model'])
        #self.opt.load_state_dict(data['opt'])
        self.ema.load_parameters(data['ema'])
        #self.ema = EMA(self.model, beta=self.ema.beta, update_every=self.ema.update_every)
        self.loss = data['loss']

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while (self.step < self.train_num_steps):
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data, _ = next(self.ds)
                    loss = self.model(data)
                    loss = (loss / self.gradient_accumulate_every)
                    total_loss += loss.item()
                    self.opt.step(loss)
                loss.sync()
                pbar.set_description(f'loss: {total_loss:.4f}')
                self.loss.append(total_loss)
                self.ema.update()
                if ((self.step != 0) and ((self.step % self.save_and_sample_every) == 0)):
                    self.ema.ema_model.eval()
                    with jt.no_grad():
                        milestone = (self.step // self.save_and_sample_every)
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map((lambda n: self.ema.ema_model.sample(batch_size=n)), batches))
                    all_images = jt.contrib.concat(all_images_list, dim=0)
                    all_images = all_images.expand(-1, 3, *all_images.shape[2:])
                    jt.save_image(all_images, str((self.results_folder / f'sample-{milestone}.png')), nrow=int(math.sqrt(self.num_samples)))
                    self.save(milestone)
                self.step += self.gradient_accumulate_every
                pbar.update(self.gradient_accumulate_every)
        print('training complete')
    
    def autodiff(self):
        from jittor_utils import auto_diff
        hook = auto_diff.Hook('diffusion')
        hook.hook_module(self.model)
        hook.hook_optimizer(self.opt)
        data, _ = next(self.ds)
        np.random.seed(0)
        data = np.random.randint(0, 256, tuple(data.shape))
        #breakpoint()
        loss = self.model(jt.float32(data))
        loss.sync()
