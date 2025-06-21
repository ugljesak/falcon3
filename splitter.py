import argparse
import os

def split_file_on_string(input_path, split_string):
    """
    Reads a file and splits its content into two parts based on a separator string.

    Args:
        input_path (str): The path to the input file.
        split_string (str): The string to split the file content on.
    """
    try:
        # Added errors='ignore' to handle non-utf-8 characters
        with open(input_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Find the position of the split string
    if split_string not in content:
        print(f"Error: The split string '{split_string}' was not found in the file.")
        return

    # Split the content into two parts
    parts = content.split(split_string, 1)
    part1_content = parts[0]
    part2_content = split_string + parts[1]

    # Create output filenames
    base_name, ext = os.path.splitext(input_path)
    output_path1 = f"{base_name}_part1{ext}"
    output_path2 = f"{base_name}_part2{ext}"

    # Write the content to the new files
    try:
        with open(output_path1, 'w', encoding='utf-8') as f:
            f.write(part1_content)
        print(f"Successfully created '{output_path1}'")

        with open(output_path2, 'w', encoding='utf-8') as f:
            f.write(part2_content)
        print(f"Successfully created '{output_path2}'")
    except Exception as e:
        print(f"An error occurred while writing the files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a file into two parts based on a specific string."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The path to the file you want to split."
    )
    args = parser.parse_args()

    # The string to find and split on
    SPLIT_MARKER = '0'

    split_file_on_string(args.input_file, SPLIT_MARKER)