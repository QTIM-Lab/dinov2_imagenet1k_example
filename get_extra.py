import argparse
from dinov2.data.datasets import ImageNet

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process ImageNet dataset.")
    parser.add_argument("--root", required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--extra", required=True, help="Path to the extra directory of the dataset")
    return parser.parse_args()

def process_dataset(root, extra):
    """
    Process the ImageNet dataset for each split and dump extra data.

    Args:
        root (str): The path to the root directory of the dataset.
        extra (str): The path to the extra directory of the dataset.
    """
    for split in ImageNet.Split:
        dataset = ImageNet(split=split, root=root, extra=extra)
        dataset.dump_extra()

def main():
    """
    Main function to process the ImageNet dataset.
    """
    args = parse_args()
    process_dataset(args.root, args.extra)

if __name__ == "__main__":
    main()
