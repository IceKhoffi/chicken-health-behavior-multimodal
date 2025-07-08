import argparse
from pathlib import Path
import random
import shutil

def main(data_path: Path, train_pct: float):
    #Validasi pada input
    if not data_path.is_dir():
        print(f'Error no directory found : {data_path}')
        return

    if not 0.01 <= train_pct <= 0.99:
        print(f'Error train_pct split need to be between 0.01 and 0.99')
        return
    
    val_pct = 1 - train_pct

    #Input Folder
    img_input = data_path / 'images'
    label_input = data_path / 'labels'

    if not img_input.exists() or not label_input.exists():
        print(f'Error required directories not found')
        return
    
    #Output Folder
    project_root = Path.cwd()
    folders = {
        'train': {
            'img': project_root / 'data' / 'train' / 'images',
            'lbl': project_root / 'data' / 'train' / 'labels'
        },
        'val': {
            'img': project_root / 'data' / 'validation' / 'images',
            'lbl': project_root / 'data' / 'validation' / 'labels'
        }
    }

    #Make Folder
    for folder in folders.values():
        folder['img'].mkdir(parents=True, exist_ok=True)
        folder['lbl'].mkdir(parents=True, exist_ok=True)

    #Copy image files
    image_files = [f for f in img_input.glob('*') if f.is_file()]
    total = len(image_files)

    if total == 0:
        print('Error no image found')
        return
    
    #Keep Image Shuffled
    random.shuffle(image_files)

    #Count Train and Validation
    train_count = int(total * train_pct)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    print(f'\nTotal images : {total}')
    print(f'Train: {len(train_files)} || Val: {len(val_files)}\n')

    def move_files(file_list, target_img: Path, target_lbl: Path):
        for img_file in file_list:
            lbl_file = label_input / f"{img_file.stem}.txt"

            #Copy image
            shutil.copy(img_file, target_img / img_file.name)

            if lbl_file.exists():
                shutil.copy(lbl_file, target_lbl / lbl_file.name)

    #Move files
    move_files(train_files, folders['train']['img'], folders['train']['lbl'])
    move_files(val_files, folders['val']['img'], folders['val']['lbl'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument('--datapath', type=Path, required=True,
                        help='Path to the dataset folder containing "images/" and "labels/" subfolers.' )
    parser.add_argument('--train_pct', type=float, default=0.8,
                        help='Proportion of data for trianing.')
    
    args = parser.parse_args()

    main(args.datapath, args.train_pct)