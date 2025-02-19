import cv2
import re
import numpy as np

################################### Detection ###################################

# Works only with the fine-tuned models like PaliGemma-3b-mix-224
# Prompt: 'Detect <entity>'
# <loc[value]> is the token used to detect objects in the image
# There are 1024 total loc tokens
# Each detection is represented by a bounding box with 4 values (in order): y_min, x_min, y_max, x_max
# To convert x values to coordinate, use the following formula: value * image_width / 1024
# To convert y values to coordinate, use the following formula: value * image_height / 1024


def display_detection(decoded, image_file_path):

    image = cv2.imread(image_file_path)
    # Get all bounding boxes and labels
    values = re.findall(r"<loc(\d+)>", decoded)
    labels = re.findall(r"\s(\w+)", decoded)

    if len(values) % 4 != 0:
        raise ValueError("Bounding box data is incomplete.")

    # Generate random colors for each box
    colors = [
        tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(values) // 4)
    ]

    height, width, _ = image.shape

    # The coordinates are always in groups of 4
    for i in range(0, len(values), 4):
        y_min = int(values[i]) * height // 1024
        x_min = int(values[i + 1]) * width // 1024
        y_max = int(values[i + 2]) * height // 1024
        x_max = int(values[i + 3]) * width // 1024

        label = labels[i // 4]
        color = colors[i // 4]

        # the order is always y_min, x_min, y_max, x_max
        overlay = image.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        alpha = 0.5  # opacity
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Bounding box outline
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Label text
        cv2.putText(
            image,
            label,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.imshow("image", image)
    cv2.waitKey(0)  # Wait for any key to close the window
    cv2.destroyAllWindows()


# test = "<loc0195><loc0015><loc0971><loc1023> tiger; <loc0289><loc0015><loc0971><loc0900> hihi"
# display_detection(
#     test,
#     "C:/Users/mlasb/Desktop/Travail 2024-2025/Projets/vlm/vision-language-model/images/tiger.jpg",
# )
