import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt


def visualize_entire_image(image_dict):
  """
    Visualize an image with bounding boxes and landmarks.

    Args:
        image_dict: A dictionary containing image details.
    """
  image = cv2.imread(os.path.join(root_dir, image_dict['filename']))
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  plt.imshow(image)

  rect = plt.Rectangle((image_dict['box_left'], image_dict['box_top']),
                image_dict['box_width'], image_dict['box_height'],
                fill=False, edgecolor='mediumseagreen', linewidth=2)
  plt.gca().add_patch(rect)

  for x, y in image_dict['landmarks']:
    plt.scatter(x, y, s=1, color='cyan', marker='o', linewidths=1)

  plt.show()


def visualize_train_data(cropped_image, landmarks):
  """
    Visualize a cropped image with bounding boxes and landmarks.

    Args:
        cropped_image: A cropped image.
        landmarks: A list of landmarks.
  """
  plt.imshow(cropped_image)

  for x, y in landmarks:
    plt.scatter(x, y, s=1, color='cyan', marker='o', linewidths=2)

  plt.show()