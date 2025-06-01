import os, zipfile, shutil

def extract_and_prepare_data(train_zip, test_zip, train_dir, test_dir):
    # If data folders already exist, skip everything
    if os.path.exists(os.path.join(train_dir, "cats")) and os.path.exists(os.path.join(test_dir, "cats")):
        print("✅ Data already prepared. Skipping extraction.")
        return

    print("🔧 Extracting and preparing data...")

    # Extract zip files
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(train_dir)
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall(test_dir)

    # Clean up nested folders in test
    nested_test = os.path.join(test_dir, 'test_set', 'test_set')
    for folder in ["cats", "dogs"]:
        shutil.move(os.path.join(nested_test, folder), test_dir)
    shutil.rmtree(os.path.join(test_dir, 'test_set'))

    # Clean up nested folders in train
    nested_train = os.path.join(train_dir, 'training_set', 'training_set')
    for folder in ["cats", "dogs"]:
        shutil.move(os.path.join(nested_train, folder), train_dir)
    shutil.rmtree(os.path.join(train_dir, 'training_set'))

    print("✅ Data extraction complete.")