"""
fix_captions.py - Convert captions file to correct format

Original format:  image.jpg#0\tCaption here
Expected format:  image.jpg,Caption here
"""

import re

input_file = "data/Flickr8k/captions.txt"
output_file = "data/Flickr8k/captions_fixed.txt"
backup_file = "data/Flickr8k/captions_original_backup.txt"

print("Converting captions file format...")

# Read original file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in original file: {len(lines)}")

# Process lines
fixed_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # Split on tab or multiple spaces
    parts = re.split(r'\t+|\s{2,}', line, maxsplit=1)
    
    if len(parts) != 2:
        continue
    
    img_id, caption = parts
    
    # Remove the #0, #1, etc. suffix from image name
    img_name = re.sub(r'#\d+$', '', img_id)
    
    # Create new format: image.jpg,caption
    fixed_line = f"{img_name},{caption}\n"
    fixed_lines.append(fixed_line)

print(f"Converted lines: {len(fixed_lines)}")

# Backup original file
print(f"Backing up original to: {backup_file}")
with open(backup_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Write fixed file
print(f"Writing fixed captions to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

# Replace original with fixed version
import shutil
shutil.copy(output_file, input_file)

print("\n✓ Captions file fixed!")
print(f"✓ Original backed up to: {backup_file}")
print(f"\nFirst 5 lines of fixed file:")

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  {i+1}. {line.strip()[:80]}")

print("\nNow run: python train.py")