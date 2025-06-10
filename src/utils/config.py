import argparse


class Config:
    """Configuration management for the project."""

    def __init__(self):
        self.parser = argparse.Argumen
        tParser(
            description='MNIST Deep Learning Classification',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._add_arguments()


    def _add_arguments(self):
        """Add command line arguments."""


        # Model architecture
        self.parser.add_argument(
            '--arch',
            default='improved',
            choices=['base', 'improved'],
            help='Model architecture to use')

        # Training parameters
        self.parser.add_argument(
            '--epochs',
            default=10,
            type=int,
            help='Number of training epochs'
        )
        self.parser.add_argument(
            '--batch_size',
            default=64,
            type=int,
            help='Batch size for training'
        )
        self.parser.add_argument(
            '--learning_rate',
            default=0.001,
            type=float,
            help='Learning rate for optimizer'
        )

        # Optimization
        self.parser.add_argument(
            '--optimizer',
            default='adamw',
            choices=['adam', 'adamw', 'sgd'],
            help='Optimizer type'
        )
        self.parser.add_argument(
            '--scheduler',
            default='onecycle',
            choices=['step', 'onecycle', 'none'],
            help='Learning rate scheduler'
        )

        # Device selection - mutually exclusive group
        device_group = self.parser.add_mutually_exclusive_group()
        device_group.add_argument(
            '--gpu',
            action='store_true',
            help='Use GPU for training (if available)'
        )
        device_group.add_argument(
            '--cpu',
            action='store_true',
            help='Force use of CPU for training'
        )
        # Data and system parameters
        self.parser.add_argument(
            '--num_workers',
            default=2,
            type=int,
            help='Number of data loading workers'
        )

        # Paths
        self.parser.add_argument(
            '--data_dir',
            default='./data',
            help='Directory for dataset'
        )
        self.parser.add_argument(
            '--save_dir',
            default='./models/',
            help='Directory to save models'
        )

        # Miscellaneous
        self.parser.add_argument(
            '--seed',
            default=42,
            type=int,
            help='Random seed for reproducibility'
        )
        self.parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )

    def parse_args(self):
         """Parse and process command line arguments."""
         args = self.parser.parse_args()

         # Handle device selection logic
         if args.cpu:
            args.use_gpu = False
         elif args.gpu:
            args.use_gpu = True
         else:
            # Default behavior: try GPU if available, fallback to CPU
            args.use_gpu = True

         # Create directories if they don't exist
         os.makedirs(args.data_dir, exist_ok=True)
         os.makedirs(args.save_dir, exist_ok=True)

         return args
