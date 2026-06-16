import os
import urllib.request
import zipfile

# Define paths
dest_dir = r"C:\Users\ISAGI\.gemini\antigravity-ide\scratch\uas_data_mining_classification"
os.makedirs(dest_dir, exist_ok=True)

zip_path = os.path.join(dest_dir, "bank_marketing.zip")
dataset_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

print(f"Downloading dataset from {dataset_url}...")
urllib.request.urlretrieve(dataset_url, zip_path)
print("Download complete.")

print("Extracting main ZIP file...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dest_dir)

# The bank marketing zip contains another zip named bank.zip and bank-additional.zip
# We want the full dataset 'bank-additional-full.csv' inside bank-additional.zip
inner_zip_path = os.path.join(dest_dir, "bank-additional.zip")
print(f"Extracting inner ZIP file: {inner_zip_path}...")
with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
    zip_ref.extractall(dest_dir)

print("Dataset extraction completed successfully!")
