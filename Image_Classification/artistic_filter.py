from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os


def apply_cartoon_filter(
    image_path: str,
    output_path: str = "cartoon_output.jpg",
) -> None:
    try:
        image = Image.open(image_path).convert("RGB")

        # Keep more detail than the original 128 x 128 blur example.
        image = image.resize((512, 512))

        # Smooth small details while preserving the main shapes.
        smoothed = image.filter(ImageFilter.MedianFilter(size=5))

        # Increase saturation for a more colorful cartoon appearance.
        color_enhancer = ImageEnhance.Color(smoothed)
        colorful = color_enhancer.enhance(1.6)

        # Increase contrast slightly.
        contrast_enhancer = ImageEnhance.Contrast(colorful)
        contrasted = contrast_enhancer.enhance(1.2)

        # Find visible outlines in the image.
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        edges = edges.point(lambda value: 255 if value > 90 else 0)
        edges = edges.convert("RGB")

        # Combine the colorful image with the dark edge outlines.
        cartoon = Image.blend(contrasted, edges, alpha=0.18)

        # Sharpen the final image.
        cartoon = cartoon.filter(ImageFilter.SHARPEN)

        cartoon.save(output_path)

        print(f"Cartoon image saved as: {output_path}")

    except Exception as error:
        print(f"Error processing image: {error}")


if __name__ == "__main__":
    print("Cartoon Image Filter")
    print("Type 'exit' to quit.\n")

    while True:
        image_path = input(
            "Enter image filename or 'exit' to quit: "
        ).strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        base_name, extension = os.path.splitext(image_path)
        output_file = f"{base_name}_cartoon{extension}"

        apply_cartoon_filter(image_path, output_file)