from PIL import Image, ImageDraw
import os

def make_seg_base(DATA_FOLDER, filename, output_path):
    try:
        # Construct the full path to the image
        image_path = os.path.join(DATA_FOLDER, filename)
        # Open the image
        with Image.open(image_path).convert('L') as image:
            
            draw = ImageDraw.Draw(image)
            # All of Image makes to 0
            draw.rectangle([0, 0, image.size[0], image.size[1]], fill=0)
            
            # Save the image to the output folder
            image.save(output_path)
    except Exception as e:
        print(f"Error processing file {filename}: {e}")


# Function to draw polygons on an image
def draw_polygon(target_label_path, points, label_value):
    try:
        with Image.open(target_label_path).convert('L') as image:
            
            draw = ImageDraw.Draw(image)
            # Draw the polygon
            draw.polygon(points, fill=label_value, width=0)    
            # Save the image to the output folder
            image.save(target_label_path)
            # print(f"Saved processed image to: {output_path}")
    except Exception as e:
        print(f"Error processing file {target_label_path}: {e}")
