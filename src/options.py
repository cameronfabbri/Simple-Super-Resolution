"""
Options for training
"""


def parse(parser, argv):
    parser.add_argument(
        '--data_dir', required=False, default='data', type=str,
        help='')
    parser.add_argument(
        '--num_workers', required=False, default=0, type=int,
        help='Number of threads for data loading class')
    parser.add_argument(
        '--patch_size', required=False, default=256, type=int,
        help='Size of the image to crop for training',
        choices=[128, 256, 512, 1024])
    parser.add_argument(
        '--batch_size', required=False, default=2, type=int,
        help='')
    parser.add_argument(
        '--num_blocks', required=False, default=8, type=int,
        help='Number of residual blocks to use')
    parser.add_argument(
        '--lambda_l1', required=False, default=100., type=float,
        help='Scalar for L1 value')
    parser.add_argument(
        '--lr_g', required=False, default=2e-4, type=float,
        help='')
    parser.add_argument(
        '--lr_d', required=False, default=2e-4, type=float,
        help='')
    parser.add_argument(
        '--num_steps', required=False, default=100000, type=int)
    return parser.parse_args(argv[1:])

