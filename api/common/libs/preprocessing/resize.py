import sys
from PIL import Image

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    aspect_ratio = width / float(height)

    if width > height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)
    padded_image = Image.new('RGB', (size, size), (255, 255, 255))
    padded_image.paste(resized_image, ((size - new_width) // 2, (size - new_height) // 2))

    padded_image.save(output_image_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python resize_image.py input_image_path output_image_path target_size")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        target_size = int(sys.argv[3])

        resize_image(input_image_path, output_image_path, target_size)
