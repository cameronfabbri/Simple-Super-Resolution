"""

Model for training a super resolution algorithm

LR: low resolution
HR: high resolution
SR: super resolved (generated images)

"""
import os

from PIL import Image

import torch
import src.data as data
import src.networks as networks
import torchvision
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.utils.data import DataLoader

opj = os.path.join


class Model:
    """ Class for training a model """

    def __init__(self, args):

        self.data_dir = args.data_dir
        self.lambda_l1 = args.lambda_l1
        self.num_steps = args.num_steps
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_blocks = args.num_blocks

        train_dir = opj(self.data_dir, 'train')
        test_dir = opj(self.data_dir, 'test')

        # Vanilla GAN loss
        self.crit = torch.nn.BCEWithLogitsLoss()

        # Pixel-wise loss for G
        self.l1_loss = torch.nn.L1Loss()

        # Set CPU/GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Get residual blocks for the generator
        resblocks = networks.SISR_Resblocks(args.num_blocks)
        self.net_g = networks.Generator(resblocks).to(self.device)

        # Size of the second to last fully connected layer in D
        fc_size = (self.patch_size // 16) **2 * 512
        self.net_d = networks.Discriminator(fc_size).to(self.device)

        self.optimizer_g = torch.optim.Adam(
            self.net_g.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(), lr=args.lr_d, betas=(0.5, 0.9))

        # Checkpoint for saving out the model
        os.makedirs('ckpts', exist_ok=True)
        self.model_path = opj('ckpts', 'model.pt')
        self.step = 0

        # Get the training dataset
        train_dataset = data.ImageDataset(root_dir=train_dir, patch_size=self.patch_size)
        self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=args.num_workers)

        # Get a batch of test images to save during training
        self.static_test_batch = []
        test_paths = [opj(test_dir, x) for x in os.listdir(test_dir)]

        num_test = 4
        to_tensor = transforms.ToTensor()
        for y_path in test_paths[:num_test]:
            y = to_tensor(Image.open(y_path))
            y = torch.unsqueeze(y, 0)
            y = (y * 2.) - 1.

            # Downsample
            resize_fn = torchvision.transforms.Resize(
                (y.shape[2] // 4, y.shape[3] //4))
            self.static_test_batch.append(
                [resize_fn(y), y])

    def save(self):
        """ Saves both networks and optimizers """
        print('Saving models...')
        self.ckpt = {
            'net_g': self.net_g.state_dict(),
            'net_d': self.net_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'step': self.step
        }
        torch.save(self.ckpt, self.model_path)
        print('Saved to', self.model_path)

    def step_g(
            self,
            batch_x: torch.Tensor,
            batch_y: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Params
        ------
        batch_x: torch.Tensor
            Batch of downsampled low resolution images
        batch_y: torch.Tensor
            Batch of high resolution versions of batch_x

        Returns
        -------
        l1_loss: torch.Tensor
            Pixel-wise L1 loss between batch_y and generated batch_g
        g_loss: torch.Tensor
            GAN loss for batch_g
        batch_g: torch.Tensor
            The batch of generated images - used in the call to step_d
        """
        self.optimizer_g.zero_grad()

        # Generate a batch of images
        batch_g = self.net_g(batch_x)

        # Pixel-wise loss
        l1_loss = self.lambda_l1 * self.l1_loss(batch_g, batch_y)

        # Send fake image to discriminator
        d_fake = self.net_d(batch_g)

        # GAN loss for G
        g_loss = self.crit(d_fake, torch.ones((d_fake.shape)))

        total_loss = l1_loss + g_loss

        total_loss.backward()

        self.optimizer_g.step()

        return l1_loss, g_loss, batch_g

    def step_d(self, batch_y: torch.Tensor, batch_g: torch.Tensor) -> torch.Tensor:
        """
        Params
        ------
        batch_y: torch.Tensor
            Batch of downsampled low resolution images
        batch_g: torch.Tensor
            Batch of generated SR images from batch_x

        Returns
        -------
        g_loss: torch.Tensor
            Loss for the discriminator
        """

        self.optimizer_d.zero_grad()

        # D's prediction on real data
        d_real = self.net_d(batch_y)

        # D's prediction on fake data
        d_fake = self.net_d(batch_g.detach())

        # Loss for D on real data
        loss_d_real = self.crit(d_real, torch.ones((d_real.shape)))

        # Loss for D on fake data
        loss_d_fake = self.crit(d_fake, torch.zeros((d_real.shape)))

        loss_d = loss_d_real + loss_d_fake

        loss_d.backward()

        self.optimizer_d.step()

        return loss_d

    def train(self, writer):
        """ Main training loop """

        for step in range(self.num_steps):

            for batch_x, batch_y in self.train_dataloader:

                # Put data on cpu/gpu
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Get loss for G and D
                l1_loss, g_loss, batch_g = self.step_g(batch_x, batch_y)
                d_loss = self.step_d(batch_y, batch_g)

                statement = '| step: ' + str(self.step)
                statement += ' | l1_loss: %.3f' % l1_loss
                statement += ' | g_loss: %.3f' % g_loss
                statement += ' | d_loss: %.3f' % d_loss
                print(statement)

                # Add loss to Tensorboard
                writer.add_scalar('l1_loss', l1_loss, self.step)
                writer.add_scalar('g_loss', g_loss, self.step)
                writer.add_scalar('d_loss', d_loss, self.step)

                self.step += 1

                # Save every 500 steps
                if not self.step % 500:

                    self.save()
                    print('Saving out test images')
                    with torch.no_grad():
                        for i, test_batch in enumerate(self.static_test_batch):

                            test_x = test_batch[0]
                            test_y = test_batch[1]

                            test_g = self.net_g(test_x)

                            # Resize x to be the same size as y for viewing,
                            # and also resize g because it could be a pixel or
                            # two off from rounding
                            resize2_fn = torchvision.transforms.Resize(
                                (test_y.shape[2], test_y.shape[3]))

                            test_x = resize2_fn(test_x)
                            test_g = resize2_fn(test_g)

                            # [-1, 1] -> [0, 1]
                            test_x = (test_x + 1.) / 2.
                            test_y = (test_y + 1.) / 2.
                            test_g = (test_g + 1.) / 2.
                            canvas = torch.cat([test_x, test_y, test_g], axis=2)
                            save_image(
                                canvas, opj('ckpts', str(self.step) + '_' + str(i) + '.png'))

