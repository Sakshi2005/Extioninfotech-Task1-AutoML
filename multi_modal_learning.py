import os
from PIL import Image, ImageDraw

def load_image(image_path):
    """Load and return an image from the specified path."""
    print(f"Checking if the image exists at: {image_path}")  # Debug print
    if os.path.isfile(image_path):
        try:
            image = Image.open(image_path)
            print("Image loaded successfully.")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        print(f"Image file not found: {image_path}")
        return None

def process_text(text):
    """Process text input for multi-modal learning."""
    print(f"Processing text: {text}")

def multi_modal_task(image_path, text):
    """Perform multi-modal learning task."""
    image = load_image(image_path)
    if image is not None:
        process_text(text)
        print("Performing multi-modal learning task...")
        # Here you could implement your multi-modal learning logic

# Example usage
if __name__ == "__main__":
    # Update this path to point to the actual image file location
    image_path = r"D:\Codesoft project sakshi\image.png"  # Ensure this path is correct
    text_data = "This is an example text for multi-modal learning."
    multi_modal_task(image_path, text_data)

def create_image(image_path):
    """Create a simple image and save it."""
    # Create a new image with white background
    width, height = 200, 100  # Size of the image
    image = Image.new("RGB", (width, height), "white")
    
    # Draw a simple rectangle (for example)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 180, 80], fill="blue")  # Draw a blue rectangle

    # Save the image to the specified path
    image.save(image_path)
    print(f"Image created and saved at: {image_path}")

# Specify the path where you want to save the image
image_path = r"D:\Codesoft project sakshi\image.png"
create_image(image_path)