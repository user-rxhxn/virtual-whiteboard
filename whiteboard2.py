import cv2
import mediapipe as mp
import numpy as np

base_options = mp.tasks.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)

camera = cv2.VideoCapture(0)

canvas = np.zeros(
    (480, 640, 3),
    dtype=np.uint8
)

previous_x = None
previous_y = None

smooth_x = None
smooth_y = None

# Fullscreen window
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Webcam",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:

    success, frame = camera.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:

        IndexTip = result.hand_landmarks[0][8]
        indexJoint = result.hand_landmarks[0][6]

        index_finger_up = IndexTip.y < indexJoint.y

        height, width, _ = frame.shape

        pixel_x = int(IndexTip.x * width)
        pixel_y = int(IndexTip.y * height)

        # Smoothing
        if smooth_x is None:
            smooth_x = pixel_x
            smooth_y = pixel_y
        else:
            smooth_x = int(
                smooth_x + (pixel_x - smooth_x) / 2
            )
            smooth_y = int(
                smooth_y + (pixel_y - smooth_y) / 2
            )

        # Draw only when index finger is up
        if index_finger_up:

            if previous_x is not None and previous_y is not None:

                cv2.line(
                    canvas,
                    (previous_x, previous_y),
                    (smooth_x, smooth_y),
                    (255, 0, 0),
                    5
                )

            previous_x = smooth_x
            previous_y = smooth_y

        else:

            previous_x = None
            previous_y = None

        # Finger tracker
        cv2.circle(
            frame,
            (smooth_x, smooth_y),
            15,
            (0, 255, 0),
            -1
        )

    combined = cv2.add(frame, canvas)

    cv2.imshow("Webcam", combined)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

camera.release()
cv2.destroyAllWindows()