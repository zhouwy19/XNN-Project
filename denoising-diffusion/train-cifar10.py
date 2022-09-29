import pathlib
import jittor as jt
from jittor import dataset
from jittor import transform as T
from denoising_diffusion_jittor import Unet, GaussianDiffusion, Trainer
import math
from PIL import Image

jt.flags.use_cuda = jt.has_cuda
jt.flags.log_silent = True

dataset_name = 'cifar10'
batch_size=128
image_size=32
channels=3
dim_mults=(1, 2, 2, 2)
train_num_steps=80*10**4
train_lr=2*10**-4

transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
ds = dataset.CIFAR10(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)

model = Unet(
    dim = batch_size,
    channels = channels,
    dim_mults = dim_mults 
)

diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l2',   # L1 or L2
    beta_schedule='linear',
    channels=channels,
)

trainer = Trainer(
    diffusion, ds, gradient_accumulate_every=10,
    train_num_steps=train_num_steps,
    save_and_sample_every=10*10**3, num_samples=64, ema_update_every=10,
    ema_decay=0.9999, train_lr=train_lr,
    results_folder=f'./results/{dataset_name}/')

# load model with name from folder
#trainer.load(name, folder)
trainer.load(milestone, model_ext='pkl')

# During training, this will also auto save milestone models and samples
# for every save_and_sample_every steps 
trainer.train()

from utils.generate_result import sample_save_many, sample_save_tile
# generate a 9x9 tile of samples
sample_save_tile(trainer, num_samples=9*9, save_dir=trainer.results_folder / 'samples/tiles/')

# generate a large set of samples for computing fid, etc.
sample_save_many(trainer, num_samples=50000, save_dir=trainer.results_folder / 'samples')
