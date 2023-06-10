import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2 
import numpy as np
from convertTransparent import convert_to_transparent
import sys
import os

if __name__ == "__main__":
    image_path = sys.argv[1]

    seg = instanceSegmentation()
    seg.load_model("pointrend_resnet50.pkl")
    seg.segmentImage(image_path, show_bboxes=True, output_image_name="POG.jpg",
    extract_segmented_objects= True, save_extracted_objects=True)

    segmask, output = seg.segmentImage("images/sample_images/sample2.jpg", extract_segmented_objects= True)

    # Specify the current file path
    current_file_path = "segmented_object_1.jpg"

    # Specify the new file path and name
    new_file_path = image_path.split("/")[-1]

    # Check if the current file exists before attempting to rename
    if os.path.exists(current_file_path):
        os.rename(current_file_path, new_file_path)

    convert_to_transparent(image_path) 

