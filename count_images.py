import os
import argparse

def count_images(root_folder):
    """
    Count the number of image files in a given folder and its subfolders.

    Args:
        root_folder (str): The path to the root folder to search for images.

    Returns:
        int: The total number of image files found.
    """
    total_images = 0
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            # Check if the file has an image extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                total_images += 1
    return total_images

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Count image files in folders.")
    parser.add_argument("--train_folder", help="Path to the train folder")
    parser.add_argument("--val_folder", help="Path to the validation folder")
    return parser.parse_args()

def main():
    """
    Main function to count images in the specified folders.
    """
    args = parse_args()

    train_images_count = count_images(args.train_folder)
    print(f"Total number of train images: {train_images_count}")

    val_images_count = count_images(args.val_folder)
    print(f"Total number of val images: {val_images_count}")

if __name__ == "__main__":
    main()
