import os

DATA_DIR = 'dataset'
total_files = 0
class_counts = {}

if not os.path.exists(DATA_DIR):
    print(f"Error: Directory not found at '{DATA_DIR}'")
else:
    print(f"--- Dataset Balance Report for '{DATA_DIR}' ---")
    try:
        for class_name in sorted(os.listdir(DATA_DIR)):
            class_path = os.path.join(DATA_DIR, class_name)
            if os.path.isdir(class_path):
                num_files = len(os.listdir(class_path))
                class_counts[class_name] = num_files
                total_files += num_files

        for class_name, count in class_counts.items():
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            print(f"- {class_name:<20}: {count} images ({percentage:.1f}%)")

        print("--------------------------------------------")
        print(f"Total Images: {total_files}")
    except Exception as e:
        print(f"An error occurred: {e}")