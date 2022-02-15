"""

"""
import sys
import argparse

import src.model as model
import src.options as options

from torch.utils.tensorboard import SummaryWriter


def print_args(parser, args):
    """ Prints out arguments passed and defaults """
    message = '\n----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------\n'
    print(message)


def main(argv):

    parser = argparse.ArgumentParser()
    args = options.parse(parser, argv)

    print_args(parser, args)

    # Writer for Tensorboard
    writer = SummaryWriter()

    sr_model = model.Model(args)
    sr_model.train(writer)


if __name__ == '__main__':
    main(sys.argv)

