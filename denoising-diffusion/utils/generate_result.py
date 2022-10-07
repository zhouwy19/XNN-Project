from pathlib import Path
from PIL import Image
import jittor as jt
import math
from tqdm.auto import tqdm

def num_to_groups(num, divisor):
    groups = (num // divisor)
    remainder = (num % divisor)
    arr = ([divisor] * groups)
    if (remainder > 0):
        arr.append(remainder)
    return arr

def batch_to_image(batch):
    batch_img = batch.clamp(0, 255).permute(0, 2, 3, 1).uint8().numpy()
    images = [Image.fromarray(img) for img in batch_img]
    return images

def sample_save_many(
        trainer,
        num_samples=50000,
        save_dir=None,
        image_ext='png'
    ):
    if save_dir is None:
        save_dir = trainer.results_folder / 'samples'
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    batch_sizes = num_to_groups(num_samples, trainer.batch_size)
    with tqdm(total=num_samples) as pbar:
        for i, size in enumerate(batch_sizes):
            count = i*trainer.batch_size
            samples = batch_to_image(trainer.model.sample(size))
            for j, im in enumerate(samples):
                num_places = math.floor(math.log10(num_samples+1))
                im.save(save_dir / f'{count+j:0{num_places}}.{image_ext}')
            pbar.update(size)

def sample_save_tile(
        trainer,
        num_samples=64,
        save_dir=None,
        image_ext='png'
    ):
    if save_dir is None:
        save_dir = train_model.results_folder / 'samples'
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # divide num_samples by batch_size
    batches = num_to_groups(num_samples, trainer.batch_size)
    all_images_list = list(map(
        (lambda n: trainer.ema.ema_model.sample(batch_size=n)),
        batches
    ))
    all_images = jt.contrib.concat(all_images_list, dim=0)
    all_images = all_images.expand(-1, 3, *all_images.shape[2:])
    jt.save_image(
        all_images,
        str(self.save_dir 
            / f'samples-tile{num_samples}x{num_samples}.{image_ext}'),
        nrow=int(math.sqrt(num_samples))
    )
