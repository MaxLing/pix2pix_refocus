# pix2pix for refocus

## Idea
Assuming defocus and focus images are from different data distribution, this project use pix2pix model with WGAN loss to achieve domain transfer.

## Run
- Download and unzip refocus data file, run train.py to get the model.  
- train.py support tensorboard visualization on losses and images, also support checkpoint saving and continuation.

## Reference
- pix2pix
- WGAN-GP
