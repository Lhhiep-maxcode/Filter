import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt


def create_an_image_dict(image):
  """
    Create a dictionary containing image details.

    Args:
        image: An XML element containing image data.

    Returns:
        A dictionary with image attributes including:
        - filename (str)
        - width (int)
        - height (int)
        - box_top (int)
        - box_left (int)
        - box_width (int)
        - box_height (int)
        - landmarks (list of tuples): Each tuple contains (x, y) coordinates of a landmark.
    """

  image_dict = {}
  image_dict['filename'] = image.attrib['file']
  image_dict['width'] = int(image.attrib['width'])
  image_dict['height'] = int(image.attrib['height'])

  box = image.find("box")
  image_dict['box_top'] = int(box.attrib['top'])
  image_dict['box_left'] = int(box.attrib['left'])
  image_dict['box_width'] = int(box.attrib['width'])
  image_dict['box_height'] = int(box.attrib['height'])

  landmarks = box.findall("part")
  landmarks_list = [(int(landmark.attrib['x']), int(landmark.attrib['y'])) for landmark in landmarks]
  image_dict['landmarks'] = landmarks_list

  return image_dict



def create_data_list(xml_link):
  """
    Create a list of dictionaries containing image details.

    Args:
        xml_link: The path to the XML file containing image data.

    Returns:
        A list of dictionaries with image attributes.
    """
  tree = ET.parse(xml_link)
  root = tree.getroot()

  images = root.find('images')
  images_list = [create_an_image_dict(image) for image in images]

  return images_list


def create_landmark_dataset(image_list):
  """
  Crop images and adjust landmarks according to the bounding boxes.
    
    Args:
        image_lsit: A list of dictionary of image details.
        
    Returns:
        cropped_images: List of cropped images.
        adjusted_landmarks: List of adjusted landmarks.
  """
  cropped_images = []
  adjusted_landmarks = []

  for image_dict in image_list:
    image = cv2.imread(os.path.join(root_dir, image_dict['filename']))
    cropped_images = image[image_dict['box_top']:(image_dict['box_top'] + image_dict['box_height']), 
                           image_dict['box_left']:(image_dict['box_left'] + image_dict['box_width'])]
    adjusted_landmark = [(x - image_dict['box_left'], y - image_dict['box_top']) for x, y in image_dict['landmarks']]

    cropped_images.append(cropped_images)
    adjusted_landmarks.append(adjusted_landmark)

  return cropped_images, adjusted_landmarks
