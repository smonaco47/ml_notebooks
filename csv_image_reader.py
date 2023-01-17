from typing import Tuple, List, Dict, Optional, Union, Callable
import os
import csv
import torchvision.datasets as datasets

    
class CSVClassImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        class_file: str,
        class_csv_label: str,
        data_csv_label: str,
        pre_transform: Callable,
        transform: Optional[Callable] = None,
    ):
        self.class_file = class_file
        self.class_csv_label = class_csv_label
        self.data_csv_label = data_csv_label
        self.new_directory = '.'
        self.pre_transform = pre_transform
        super().__init__(
            root,
            transform=transform,
        )
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        class_set = set()
        with open(self.class_file) as class_file:
            reader = csv.DictReader(class_file)
            for row in reader:
                class_set.add(row[self.class_csv_label])

        classes = list(class_set)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return list(classes), class_to_idx
    
    
    def make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        instances = []
        with open(self.class_file) as class_file:
            reader = csv.DictReader(class_file)
            for row in reader:
                image_id = row[self.data_csv_label]
                original_file = os.path.join(directory, f"{image_id}.tif")
                new_file  = os.path.join(self.new_directory, f"{image_id}.jpg")
                if not os.path.exists(new_file):
                    self.pre_transform(original_file, new_file)
                item = new_file, class_to_idx[row[self.class_csv_label]]
                instances.append(item)
        return instances
