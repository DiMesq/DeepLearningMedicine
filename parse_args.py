import argparse

def parse_args():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='PCAM')
    parser.add_argument('--model-name', '-m', dest='model_name', required=True, help='Model name (for loading custom model or for transfer learning)')
    parser.add_argument('--test-run', action='store_true', help='if making a test run (uses smaller dataset)')
    parser.add_argument('--local', action='store_true', help='if running locally')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='a help')
    parser_train.set_defaults(kind="train")
    parser_train.add_argument('--resume-training', '-C', dest='resume_training', help='Run id of the model to continue')
    parser_train.add_argument('--num-epochs', '-E', dest='num_epochs', default=100, help='Model name (for loading custom model or for transfer learning)')
    parser_train.add_argument('--max-stale', '-S', dest='max_stale', default=10, type=int, help='Early stopping: number of epochs without improvement before stopping')
    parser_train.add_argument('--negative-only', action='store_true', dest='negative_only', help='only use negative data (for debugging purposes')
    parser_train.add_argument('--pretrained', action='store_true', help='uses transfer learning?')
    parser_train.add_argument('--augmentation-level', '-A', dest='augmentation_level',
                              choices=['high', 'low'], default='high',
                              help="Amount of data augmentations to use during training")

    parser_eval = subparsers.add_parser('evaluate', help='b help')
    parser_eval.set_defaults(kind="eval")
    parser_eval.add_argument('--run-id', '-ID', required=True, dest='run_id', help='Run id of the model to load')
    parser_eval.add_argument('--evaluation-kind', '-k', required=True, dest='evaluation_kind', choices=['val', 'test'], help="Either 'val' or 'test'")

    return vars(parser.parse_args())
