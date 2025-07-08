import yaml
from pathlib import Path
import argparse

def create_data_yaml(path_to_classes_txt: Path, path_to_data_yaml: Path):
    #Validate classes.txt
    if not path_to_classes_txt.exists():
        print(f"Error File '{path_to_classes_txt}' not Found.")
        return
    
    #Read Class
    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    if not classes:
        print(f"Error File classes.txt not valid.")
        return
    
    number_of_classes = len(classes)

    #Configure data.yaml
    data = {
        'path': str(path_to_data_yaml.parent.resolve()),
        'train': 'data/train/images',
        'val': 'data/validation/images',
        'nc': number_of_classes,
        'names': classes
    }

    path_to_data_yaml.parent.mkdir(parents=True, exist_ok=True)

    #Write Configure into YAML
    with open(path_to_data_yaml, 'w') as yaml_file:
        yaml.dump(data, yaml_file, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data.yaml file to configure YOLO training")
    parser.add_argument("--classes", type=str, required=True, help="Path to file classes.txt")
    parser.add_argument("--output", type=str, default="data.yaml", help="Output Path to file data.yaml")

    args = parser.parse_args()

    path_to_classes_txt = Path(args.classes)
    path_to_data_yaml = Path(args.output)

    create_data_yaml(path_to_classes_txt, path_to_data_yaml)