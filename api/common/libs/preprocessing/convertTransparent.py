from PIL import Image
import os

def convert_to_transparent(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to RGBA mode (with alpha channel)
    image = image.convert("RGBA")

    # Get the pixel data
    pixel_data = image.load()

    # Iterate over each pixel
    width, height = image.size
    for x in range(width):
        for y in range(height):
            # Check if the pixel is white
            if pixel_data[x, y] == (0, 0, 0, 255):
                # Set the pixel to be transparent
                pixel_data[x, y] = (0, 0, 0, 0)

    width, height = image.size
    for x in range(width):
        for y in range(height):
            # Check if the pixel is white
            if pixel_data[x, y] == (255, 255, 255, 255):
                # Set the pixel to be transparent
                pixel_data[x, y] = (255, 255, 255, 0)


    output_path = "transparent_images/" + image_path.split("/")[-1].split(".")[0] + "_transparent.png"
    image.save(output_path)

