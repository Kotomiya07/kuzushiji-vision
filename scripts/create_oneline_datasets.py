import ast
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, UnidentifiedImageError

# --- Configuration ---
CSV_FILE_PATH = "data/processed/merged_annotations.csv"
BASE_OUTPUT_DIR = "data/oneline"
# Assuming original_image path like: 'data/raw/dataset/BOOK_ID/images/image_file.jpg'
# BOOK_ID will be extracted from this path.
# For example, if path is 'data/raw/dataset/100249416/images/foo.jpg', BOOK_ID is '100249416'
ORIGINAL_IMAGE_PREFIX_TO_STRIP = "data/raw/dataset/"
# 実行する最大のワーカープロセス数を指定します。Noneの場合、os.cpu_count() が使われます。
MAX_WORKERS = None

# --- Helper Functions ---


def parse_box_coordinates(box_str):
    """
    Parses a string like "[163, 410, 331, 2841]" into a tuple (x_min, y_min, x_max, y_max).
    Assumes the coordinates are [x_min, y_min, x_max, y_max].
    Pillow crop expects (left, upper, right, lower).
    """
    try:
        coords = ast.literal_eval(box_str)
        if len(coords) == 4:
            # Assuming [x_min, y_min, x_max, y_max] from observation.
            # Pillow's crop uses (left, upper, right, lower)
            # If box_str was [x,y,width,height], it would be (x, y, x + width, y + height)
            return int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        else:
            # print(f"Warning: Unexpected coordinate length for {box_str}: {coords}") # Cannot print from worker easily
            return None
    except (ValueError, SyntaxError):
        # print(f"Warning: Could not parse box coordinates '{box_str}': {e}")
        return None


def decode_unicode_ids(unicode_ids_str):
    """
    Parses a string like "['U+306F', 'U+306A']" into a text string.
    """
    try:
        unicode_list = ast.literal_eval(unicode_ids_str)
        chars = []
        for unicode_id in unicode_list:
            if unicode_id.startswith("U+"):
                try:
                    char_code = int(unicode_id[2:], 16)
                    chars.append(chr(char_code))
                except ValueError:
                    # print(f"Warning: Invalid Unicode ID format in '{unicode_id}' from '{unicode_ids_str}'")
                    pass  # Silently skip problematic characters
            # else:
            # print(f"Warning: Unicode ID '{unicode_id}' does not start with 'U+' in '{unicode_ids_str}'")
        return "".join(chars)
    except (ValueError, SyntaxError):
        # print(f"Warning: Could not parse unicode_ids string '{unicode_ids_str}': {e}")
        return ""


def get_book_id_from_path(original_image_path, prefix_to_strip):
    """
    Extracts book_id from original_image_path.
    Example: 'data/raw/dataset/100249416/images/100249416_00002_2.jpg' -> '100249416'
    """
    if original_image_path.startswith(prefix_to_strip):
        path_parts = original_image_path[len(prefix_to_strip) :].split(os.sep)
        if len(path_parts) > 0:
            return path_parts[0]
    # Fallback if pattern doesn't match, try to get parent of 'images' dir
    try:
        # data/raw/dataset/BOOK_ID/images/file.jpg
        #                      ^ (this one)
        return os.path.basename(os.path.dirname(os.path.dirname(original_image_path)))
    except Exception:
        # print(f"Warning: Could not determine book_id from path: {original_image_path}")
        return "unknown_book"


# --- Worker Function for Parallel Processing ---
def process_row(row_data, row_index_1_based, base_output_dir, original_image_prefix_to_strip):
    """
    Processes a single row from the CSV file.
    Returns a tuple: (bool_success, message_or_none)
    """
    try:
        original_image_path = row_data.get("original_image", "").strip()
        box_str = row_data.get("box_in_original", "").strip()
        unicode_ids_str = row_data.get("unicode_ids", "").strip()

        if not all([original_image_path, box_str, unicode_ids_str]):
            return False, f"Row {row_index_1_based} is missing one or more required fields."

        book_id = get_book_id_from_path(original_image_path, original_image_prefix_to_strip)
        book_image_dir = os.path.join(base_output_dir, book_id, "images")
        book_text_dir = os.path.join(base_output_dir, book_id, "texts")

        # makedirs might be called concurrently, but exist_ok=True handles it.
        os.makedirs(book_image_dir, exist_ok=True)
        os.makedirs(book_text_dir, exist_ok=True)

        coords = parse_box_coordinates(box_str)
        if coords is None:
            return False, f"Row {row_index_1_based}: Could not parse coordinates '{box_str}'."

        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            return False, f"Row {row_index_1_based}: Invalid coordinates {coords} (x_max <= x_min or y_max <= y_min)."

        text_content = decode_unicode_ids(unicode_ids_str)
        if not text_content:
            return False, f"Row {row_index_1_based}: Decoded text is empty for '{unicode_ids_str}'."

        original_image_basename = os.path.splitext(os.path.basename(original_image_path))[0]
        unique_id = f"{original_image_basename}_line_{row_index_1_based}"
        image_output_path = os.path.join(book_image_dir, f"{unique_id}.jpg")

        try:
            with Image.open(original_image_path) as img:
                img_width, img_height = img.size
                clamped_coords = (max(0, coords[0]), max(0, coords[1]), min(img_width, coords[2]), min(img_height, coords[3]))
                if clamped_coords[2] <= clamped_coords[0] or clamped_coords[3] <= clamped_coords[1]:
                    return (
                        False,
                        f"Row {row_index_1_based}: Coordinates {coords} became invalid after clamping to image size {img.size} -> {clamped_coords}.",
                    )

                cropped_image = img.crop(clamped_coords)
                cropped_image.convert("RGB").save(image_output_path, "JPEG")
        except FileNotFoundError:
            return False, f"Row {row_index_1_based}: Original image not found at '{original_image_path}'."
        except UnidentifiedImageError:
            return False, f"Row {row_index_1_based}: Cannot identify image file '{original_image_path}'."
        except Exception as e:
            return False, f"Row {row_index_1_based}: Error processing image '{original_image_path}': {e}."

        text_output_path = os.path.join(book_text_dir, f"{unique_id}.txt")
        with open(text_output_path, "w", encoding="utf-8") as outfile:
            outfile.write(text_content)

        return True, None  # Success

    except Exception as e:
        # Catch any other unexpected error during row processing
        return False, f"Error processing row {row_index_1_based} (Content: {row_data}): {e}"


# --- Main Script ---
def create_oneline_dataset():
    print(f"Starting dataset creation from: {CSV_FILE_PATH}")
    print(f"Output will be saved to: {BASE_OUTPUT_DIR}")
    if MAX_WORKERS:
        print(f"Using up to {MAX_WORKERS} worker processes.")
    else:
        print(f"Using up to {os.cpu_count()} worker processes (system default).")

    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Created base output directory: {BASE_OUTPUT_DIR}")

    processed_count = 0
    error_count = 0

    # Read all rows into memory first to avoid issues with CSV reader and multiprocessing
    # For very large CSVs, this might be an issue, but typical annotation files should be manageable.
    rows_to_process = []
    try:
        with open(CSV_FILE_PATH, encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            rows_to_process = list(reader)
        print(f"Loaded {len(rows_to_process)} rows from CSV.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading CSV: {e}")
        return

    if not rows_to_process:
        print("No rows to process in the CSV file.")
        return

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_row_index = {
            executor.submit(process_row, row, i + 1, BASE_OUTPUT_DIR, ORIGINAL_IMAGE_PREFIX_TO_STRIP): i + 1
            for i, row in enumerate(rows_to_process)
        }

        print(f"Submitted {len(future_to_row_index)} tasks to process pool.")

        for future in as_completed(future_to_row_index):
            row_idx = future_to_row_index[future]
            try:
                success, message = future.result()
                if success:
                    processed_count += 1
                else:
                    error_count += 1
                    if (
                        message
                    ):  # message might be None if an unexpected exception was caught by the outer try-except in process_row
                        print(f"Warning: {message}")
            except Exception as exc:
                error_count += 1
                # This catches errors from the process_row function itself if it fails catastrophically
                # or if there's an issue with the future object.
                print(f"Error processing row index {row_idx} (future generated an exception): {exc}")

            # Progress reporting
            total_done = processed_count + error_count
            if total_done % 100 == 0 or total_done == len(rows_to_process):
                print(
                    f"Completed {total_done}/{len(rows_to_process)} tasks. (Successful: {processed_count}, Errors: {error_count})"
                )

    print("--- Dataset Creation Complete ---")
    print(f"Successfully processed: {processed_count} lines.")
    print(f"Skipped due to errors/warnings: {error_count} lines.")
    print(f"Output saved in: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    create_oneline_dataset()
