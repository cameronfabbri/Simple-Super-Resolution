"""
Script for running training
"""
import click

from torch.utils.tensorboard import SummaryWriter

import src.model as model


@click.command()
@click.option(
    '--data_dir', default='data', type=str, help='Directory for the data')
@click.option(
    '--num_workers', default=0, type=int, help='Number of threads for data loading class')
@click.option(
    '--patch_size', default=256, type=click.Choice([128, 256, 512, 1024]), help='Size of the image to crop for training')
@click.option(
    '--batch_size', default=2, type=int, help='Batch size for training')
@click.option(
    '--num_blocks', default=8, type=int, help='Number of residual blocks to use')
@click.option(
    '--lambda_l1', default=100.0, type=float, help='Scalar for L1 value')
@click.option(
    '--lr_g', default=2e-4, type=float, help='Learning rate for generator')
@click.option(
    '--lr_d', default=2e-4, type=float, help='Learning rate for discriminator')
@click.option(
    '--num_steps', default=100000, type=int, help='Number of steps for training')
def main(
        data_dir: str,
        num_workers: int,
        patch_size: int,
        batch_size: int,
        num_blocks: int,
        lambda_l1: float,
        lr_g: float,
        lr_d: float,
        num_steps: int) -> None:

    print(f'data_dir: {data_dir}')
    print(f'num_workers: {num_workers}')
    print(f'patch_size: {patch_size}')
    print(f'batch_size: {batch_size}')
    print(f'num_blocks: {num_blocks}')
    print(f'lambda_l1: {lambda_l1}')
    print(f'lr_g: {lr_g}')
    print(f'lr_d: {lr_d}')
    print(f'num_steps: {num_steps}')

    # Writer for Tensorboard
    writer = SummaryWriter()

    sr_model = model.Model(
        data_dir,
        num_workers,
        patch_size,
        batch_size,
        num_blocks,
        lambda_l1,
        lr_g,
        lr_d,
        num_steps
    )

    sr_model.train(writer)


if __name__ == '__main__':
    main()

