#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt


class TomatoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model = torch.hub.load('ultralytics/yolov5', 'custom', path="yolov5n_tomatoes_model.pt")

    def forward(self, image):
        return self._model(image)


def get_center(a, b, llimit=0, ulimit=480):
    center = int(0.5 * (a + b))
    return max(llimit, min(ulimit - 1, center))


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 2  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# framrs.align allows us to perform alignment of depth frames to others es
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

model = TomatoModel()
model.conf = 0.2

# Streaming loop
count = 0
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        frames.get_depth_frame()  # is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Массив расстояний
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # print("depth image", depth_image)
        # print("depth means", depth_image.mean(-1))

        # print("depth", depth_image.shape)
        color_image = np.asanyarray(color_frame.get_data())
        # print("color", color_image.shape)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # print("depth_image_3d", depth_image_3d)
        # depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        bg_removed = color_image
        # bg_removed = np.where((depth_image > clipping_distance), grey_color, color_image)
        count = count + 1
        # print(type(bg_removed))
        if count % 1 == 0:
            # cv2.imshow("Chto nado", bg_removed)
            # print("color_img", color_image.shape)
            # print("RGB vector", bg_removed)
            im = Image.fromarray(bg_removed)
            results = model(im)
            # print(results.xyxy[0])
            final_image = bg_removed
            int_tensor = results.xyxy[0].int()

            for box in int_tensor:
                # x_center = int(0.5 * (box[0] + box[2]))2
                # y_center = int(0.5 * (box[1] + box[3]))
                # x_center = max(min(x_center, 480 - 1), 0)
                # y_center = max(min(y_center, 640 - 1), 0)

                x_center = get_center(box[0], box[2], llimit=0, ulimit=480)
                y_center = get_center(box[1], box[3], llimit=0, ulimit=640)
                print(x_center, y_center)
                print(depth_image[x_center, y_center])
                if depth_image[x_center, y_center] > clipping_distance:
                    continue
                else:
                    final_image = cv2.rectangle(
                        final_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (255, 0, 0), 3
                    )

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        # выравнивание по прямой
        # images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            bg_removed,
            f"NUMBER OF TOMATOES: {len(int_tensor)}",
            (50, 50), font,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2
        )
        cv2.imshow('Align Example', bg_removed)
        # cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
