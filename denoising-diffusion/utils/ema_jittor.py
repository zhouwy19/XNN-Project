"""
jittor fork of https://github.com/lucidrains/ema-pytorch
maintained by me at https://github.com/alexfanqi/ema-jittor
"""

import copy
import jittor
from jittor import nn

def exists(val):
    return (val is not None)

def is_float_dtype(dtype):
    return any([(dtype == float_dtype) for float_dtype in (jittor.float64, jittor.float32, jittor.float16)])

def clamp(value, min_value=None, max_value=None):
    assert (exists(min_value) or exists(max_value))
    if exists(min_value):
        value = max(value, min_value)
    if exists(max_value):
        value = min(value, max_value)
    return value

class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        ema_model=None,
        beta=0.9999,
        update_after_step=100,
        update_every=10,
        inv_gamma=1.0,
        power=(2 / 3),
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
        ignore_names=set()
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = ema_model
        # is_stop_grad property is lost here with copy
        # but it will be corrected later with copy_params_from_model_to_ema()
        if (not exists(self.ema_model)):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()
        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema
        self.ignore_names = ignore_names
        self.initted=jittor.Var([False]).stop_grad()
        self.step=jittor.Var([0]).stop_grad()

    def copy_params_from_model_to_ema(self):
        for (ma_params, current_params) in zip(list(self.ema_model.parameters()), list(self.online_model.parameters())):
            if (not is_float_dtype(current_params.dtype)):
                continue
            if (current_params.is_stop_grad()):
                ma_params.assign(current_params.copy()).stop_grad()
            else:
                ma_params.assign(current_params.copy()).start_grad()

    def get_current_decay(self):
        epoch = clamp(((self.step.item() - self.update_after_step) - 1), min_value=0.0)
        value = (1 - ((1 + (epoch / self.inv_gamma)) ** (- self.power)))
        if (epoch <= 0):
            return 0.0
        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1
        if ((step % self.update_every) != 0):
            return
        if (step <= self.update_after_step):
            self.copy_params_from_model_to_ema()
            return
        if (not self.initted.item()):
            self.copy_params_from_model_to_ema()
            self.initted.assign(jittor.Var([True]))
        self.update_moving_average(self.ema_model, self.online_model)

    @jittor.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()
        for ((name, current_params), (_, ma_params)) in zip(list(current_model.named_parameters()), list(ma_model.named_parameters())):
            if (name in self.ignore_names):
                continue
            if (not is_float_dtype(current_params.dtype)):
                continue
            if (name in self.param_or_buffer_names_no_ema):
                if (current_params.is_stop_grad()):
                    ma_params.assign(current_params.copy()).stop_grad()
                else:
                    ma_params.assign(current_params.copy()).start_grad()
                continue
            difference = (ma_params - current_params)
            difference.assign(difference * (1.0 - current_decay))
            if (current_params.is_stop_grad()):
                ma_params.assign(ma_params - difference).stop_grad()
            else:
                ma_params.assign(ma_params - difference).start_grad()

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
