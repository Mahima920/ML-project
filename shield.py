# ============================================================
# DR. STRANGE SHIELDS - GESTURE CONTROL SYSTEM
# ============================================================
# SELFIE CAMERA VERSION
#
# Camera:
#   - Left/right mirrored like a phone selfie camera
#   - NOT upside down
#
# Gestures:
#   KEY_1 -> KEY_2 -> KEY_3 = SHIELDS ON
#   KEY_4                  = SHIELDS OFF
#
# Keyboard:
#   1 -> KEY_1
#   2 -> KEY_2
#   3 -> SHIELDS ON
#   4 -> SHIELDS OFF
#   q -> QUIT
#
# Nothing is written on the camera image.
# ============================================================


# ============================================================
# LIBRARIES
# ============================================================

import cv2
import time
import mediapipe as mp
import numpy as np
import os
import pickle
import signal
import sys

from datetime import datetime
from argparse import ArgumentParser

from utils import (
    mediapipe_detection,
    get_center_lh,
    get_center_rh,
    points_detection_hands
)

import pyvirtualcam
from pyvirtualcam import PixelFormat


# ============================================================
# INPUT PARAMETERS
# ============================================================

parser = ArgumentParser(
    description="Dr. Strange Shields - Gesture Control System"
)

parser.add_argument(
    "-m",
    "--model",
    dest="ML_model",
    default="models/model_svm.sav",
    help="PATH of model FILE.",
    metavar="FILE"
)

parser.add_argument(
    "-t",
    "--threshold",
    dest="threshold_prediction",
    default=0.85,
    type=float,
    help="Prediction confidence threshold."
)

parser.add_argument(
    "-dc",
    "--det_conf",
    dest="min_detection_confidence",
    default=0.5,
    type=float,
    help="Minimum detection confidence."
)

parser.add_argument(
    "-tc",
    "--trk_conf",
    dest="min_tracking_confidence",
    default=0.5,
    type=float,
    help="Minimum tracking confidence."
)

parser.add_argument(
    "-c",
    "--camera_id",
    dest="camera",
    default=0,
    type=int,
    help="Camera ID."
)

parser.add_argument(
    "-s",
    "--shield",
    dest="shield_video",
    default="effects/shield.mp4",
    help="PATH of shield video FILE.",
    metavar="FILE"
)

parser.add_argument(
    "-o",
    "--output",
    dest="output_mode",
    default="window",
    choices=["window", "virtual", "both"],
    help="Output mode."
)

args = parser.parse_args()


# ============================================================
# GLOBAL VARIABLES
# ============================================================

cap = None
cam = None
show_window = False


# ============================================================
# CLEAN EXIT
# ============================================================

def cleanup():

    global cap
    global cam

    print("\n")
    print("=" * 60)
    print("🧹 CLEANING UP")
    print("=" * 60)

    if cap is not None:

        try:
            cap.release()
            print("✅ Camera released")
        except Exception:
            pass

    if show_window:

        try:
            cv2.destroyAllWindows()
            print("✅ OpenCV windows closed")
        except Exception:
            pass

    if cam is not None:

        try:
            cam.close()
            print("✅ Virtual camera closed")
        except Exception:
            pass

    print("\n🏁 Dr. Strange Shields stopped.")
    print("=" * 60)


def signal_handler(sig, frame):

    print("\n")
    print("🛑 Ctrl+C received.")

    cleanup()

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

current_directory = os.path.dirname(
    os.path.realpath(__file__)
)


# ============================================================
# FILE PATHS
# ============================================================

model_path = os.path.join(
    current_directory,
    args.ML_model
)

shield_path = os.path.join(
    current_directory,
    args.shield_video
)


# ============================================================
# START CAMERA
# ============================================================

print("\n")
print("=" * 60)
print("📷 STARTING CAMERA")
print("=" * 60)

print(f"Camera ID: {args.camera}")

cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():

    print("\n❌ ERROR: Camera could not be opened.")
    print("Try another camera ID, for example:")
    print("python main.py -c 1")

    sys.exit(1)

print("✅ Camera opened successfully")


# ============================================================
# CAMERA SETTINGS
# ============================================================

# Try to use a stable resolution.
# If your webcam does not support this resolution,
# OpenCV will use the closest available resolution.

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Small delay to allow the webcam to initialise.
time.sleep(2)


# ============================================================
# GET ACTUAL CAMERA RESOLUTION
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

if width <= 0 or height <= 0:

    print("⚠️ Could not determine camera resolution.")

    width = 1280
    height = 720

print(
    f"📐 Camera resolution: {width} x {height}"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

print("\n")
print("=" * 60)
print("🤖 LOADING ML MODEL")
print("=" * 60)

print(model_path)

try:

    with open(model_path, "rb") as f:

        model = pickle.load(f)

    labels = np.array(
        model.classes_
    )

    print("✅ Model loaded successfully")

    print(
        "🏷️ Classes:",
        list(labels)
    )

except Exception as e:

    print("\n❌ ERROR: Could not load ML model.")

    print("Reason:")
    print(e)

    cap.release()

    sys.exit(1)


# ============================================================
# LOAD SHIELD VIDEO
# ============================================================

print("\n")
print("=" * 60)
print("🎬 LOADING SHIELD VIDEO")
print("=" * 60)

print(shield_path)

shield = cv2.VideoCapture(
    shield_path
)

if not shield.isOpened():

    print("\n❌ ERROR: Shield video could not be opened.")

    cap.release()

    sys.exit(1)

print("✅ Shield video loaded successfully")


# ============================================================
# INITIAL STATES
# ============================================================

KEY_1 = False
KEY_2 = False
KEY_3 = False

SHIELDS = False


# ============================================================
# GESTURE TIMERS
# ============================================================

t1 = None
t2 = None
t3 = None


# ============================================================
# GESTURE SETTINGS
# ============================================================

# Maximum time allowed between gestures.

GESTURE_TIMEOUT = 2.0


# Minimum prediction confidence.

PREDICTION_THRESHOLD = args.threshold_prediction


# ============================================================
# SHIELD SIZE
# ============================================================

scale = 1.5


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# OUTPUT SETTINGS
# ============================================================

show_window = (
    args.output_mode in
    ["window", "both"]
)

use_virtual_cam = (
    args.output_mode in
    ["virtual", "both"]
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("🛡️ DR. STRANGE SHIELDS")
print("🎯 GESTURE CONTROL SYSTEM")
print("=" * 60)

print()
print(f"📷 Camera ID: {args.camera}")

print(
    f"🤖 Model: {args.ML_model}"
)

print(
    f"🎯 Prediction threshold: "
    f"{PREDICTION_THRESHOLD}"
)

print(
    f"🔍 Detection confidence: "
    f"{args.min_detection_confidence}"
)

print(
    f"📊 Tracking confidence: "
    f"{args.min_tracking_confidence}"
)

print(
    f"🎬 Shield video: "
    f"{args.shield_video}"
)

print(
    f"📺 Output mode: "
    f"{args.output_mode.upper()}"
)


# ============================================================
# SELFIE CAMERA INFORMATION
# ============================================================

print()
print("=" * 60)
print("🤳 SELFIE CAMERA MODE")
print("=" * 60)

print(
    "✅ Left/right mirror enabled"
)

print(
    "✅ Camera will NOT be flipped vertically"
)

print(
    "✅ Camera will NOT be upside down"
)

print("=" * 60)


# ============================================================
# MEDIAPIPE SYSTEM
# ============================================================

with mp_holistic.Holistic(

    min_detection_confidence=
        args.min_detection_confidence,

    min_tracking_confidence=
        args.min_tracking_confidence,

    model_complexity=0

) as holistic:


    # ========================================================
    # VIRTUAL CAMERA
    # ========================================================

    if use_virtual_cam:

        print("\n")
        print("=" * 60)
        print("🎥 STARTING VIRTUAL CAMERA")
        print("=" * 60)

        try:

            cam = pyvirtualcam.Camera(

                width=width,

                height=height,

                fps=30,

                fmt=PixelFormat.BGR

            )

            print(
                f"✅ Virtual camera: "
                f"{cam.device}"
            )

        except Exception as e:

            print(
                "\n❌ Could not start "
                "virtual camera."
            )

            print(
                "Reason:",
                e
            )

            cam = None

            use_virtual_cam = False


    # ========================================================
    # READY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("🚀 SYSTEM READY")
    print("=" * 60)

    print()
    print("GESTURE CONTROL:")
    print()
    print("  KEY_1 → KEY_2 → KEY_3")
    print()
    print(
        f"  Each step must be completed "
        f"within {GESTURE_TIMEOUT} seconds."
    )

    print()
    print("SHIELD:")
    print()
    print("  🛡️ ON  → remains ON")
    print("  🛡️ OFF → KEY_4")

    print()
    print("KEYBOARD TEST:")
    print()
    print("  1 → KEY_1")
    print("  2 → KEY_2")
    print("  3 → SHIELDS ON")
    print("  4 → SHIELDS OFF")
    print("  q → QUIT")

    print()
    print("⚠️ Keyboard controls require")
    print("   the OpenCV window to be focused.")

    print()
    print("=" * 60)
    print()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while cap.isOpened():


            # =================================================
            # READ CAMERA
            # =================================================

            ret, frame = cap.read()

            if not ret:

                print(
                    "\n❌ Could not read "
                    "camera frame."
                )

                break


            # =================================================
            # SELFIE CAMERA
            # =================================================
            #
            # IMPORTANT:
            #
            # cv2.flip(frame, 1)
            #
            # means:
            #
            # 1 = horizontal flip
            #
            # This creates a normal selfie-style
            # mirror image.
            #
            # It does NOT flip the image upside down.
            #
            # =================================================

            frame = cv2.flip(
                frame,
                1
            )


            # =================================================
            # READ SHIELD VIDEO
            # =================================================

            ret_shield, frame_shield = (
                shield.read()
            )


            # =================================================
            # LOOP SHIELD VIDEO
            # =================================================

            if not ret_shield:

                shield.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                ret_shield, frame_shield = (
                    shield.read()
                )


            # =================================================
            # FALLBACK IF SHIELD VIDEO FAILS
            # =================================================

            if not ret_shield:

                frame_shield = np.zeros(
                    (
                        height,
                        width,
                        3
                    ),
                    dtype=np.uint8
                )


            # =================================================
            # MEDIAPIPE
            # =================================================

            frame, results = (
                mediapipe_detection(
                    frame,
                    holistic
                )
            )


            # =================================================
            # GET LEFT HAND
            # =================================================

            (
                xMinL,
                xMaxL,
                yMinL,
                yMaxL
            ) = get_center_lh(
                frame,
                results
            )


            # =================================================
            # GET RIGHT HAND
            # =================================================

            (
                xMinR,
                xMaxR,
                yMinR,
                yMaxR
            ) = get_center_rh(
                frame,
                results
            )


            # =================================================
            # SHIELD TRANSPARENCY
            # =================================================

            black_screen = np.array(
                [0, 0, 0],
                dtype=np.uint8
            )


            mask = cv2.inRange(

                frame_shield,

                black_screen,

                black_screen

            )


            res = cv2.bitwise_and(

                frame_shield,

                frame_shield,

                mask=mask

            )


            res = (
                frame_shield -
                res
            )


            # =================================================
            # SHIELD ALPHA
            # =================================================

            alpha = 1.0


            # =================================================
            # LEFT SHIELD
            # =================================================

            if (
                SHIELDS
                and xMinL is not None
                and xMaxL is not None
                and yMinL is not None
                and yMaxL is not None
            ):


                # -------------------------------------------------
                # LEFT HAND CENTER
                # -------------------------------------------------

                xc_lh = (
                    xMaxL + xMinL
                ) / 2

                yc_lh = (
                    yMaxL + yMinL
                ) / 2


                xc_lh = int(
                    width * xc_lh
                )

                yc_lh = int(
                    height * yc_lh
                )


                # -------------------------------------------------
                # SHIELD SIZE
                # -------------------------------------------------

                l_width_shield = int(

                    width
                    * (xMaxL - xMinL)
                    / 2
                    * 3.5
                    * scale

                )


                l_height_shield = int(

                    height
                    * (yMaxL - yMinL)
                    / 2
                    * 2
                    * scale

                )


                if (
                    l_width_shield > 0
                    and
                    l_height_shield > 0
                ):


                    # ---------------------------------------------
                    # RESIZE SHIELD
                    # ---------------------------------------------

                    res2 = cv2.resize(

                        res,

                        (
                            l_width_shield * 2,
                            l_height_shield * 2
                        )

                    )


                    # ---------------------------------------------
                    # INITIAL CROP
                    # ---------------------------------------------

                    start_h = 0
                    start_w = 0

                    stop_h = (
                        l_height_shield * 2
                    )

                    stop_w = (
                        l_width_shield * 2
                    )


                    # ---------------------------------------------
                    # TARGET POSITION
                    # ---------------------------------------------

                    f_start_h = (
                        yc_lh -
                        l_height_shield
                    )

                    f_stop_h = (
                        yc_lh +
                        l_height_shield
                    )

                    f_start_w = (
                        xc_lh -
                        l_width_shield
                    )

                    f_stop_w = (
                        xc_lh +
                        l_width_shield
                    )


                    # ---------------------------------------------
                    # TOP
                    # ---------------------------------------------

                    if f_start_h < 0:

                        start_h = -f_start_h

                        f_start_h = 0


                    # ---------------------------------------------
                    # BOTTOM
                    # ---------------------------------------------

                    if f_stop_h > height:

                        stop_h = (
                            res2.shape[0]
                            -
                            (
                                f_stop_h
                                -
                                height
                            )
                        )

                        f_stop_h = height


                    # ---------------------------------------------
                    # LEFT
                    # ---------------------------------------------

                    if f_start_w < 0:

                        start_w = -f_start_w

                        f_start_w = 0


                    # ---------------------------------------------
                    # RIGHT
                    # ---------------------------------------------

                    if f_stop_w > width:

                        stop_w = (
                            res2.shape[1]
                            -
                            (
                                f_stop_w
                                -
                                width
                            )
                        )

                        f_stop_w = width


                    # ---------------------------------------------
                    # CROP
                    # ---------------------------------------------

                    res2 = res2[
                        start_h:stop_h,
                        start_w:stop_w
                    ]


                    # ---------------------------------------------
                    # TARGET
                    # ---------------------------------------------

                    target = frame[
                        f_start_h:f_stop_h,
                        f_start_w:f_stop_w
                    ]


                    # ---------------------------------------------
                    # BLEND
                    # ---------------------------------------------

                    if (

                        target.size > 0

                        and

                        res2.size > 0

                        and

                        target.shape ==
                        res2.shape

                    ):

                        blended = cv2.addWeighted(

                            target,

                            alpha,

                            res2,

                            1.0,

                            1

                        )


                        frame[
                            f_start_h:f_stop_h,
                            f_start_w:f_stop_w
                        ] = blended


            # =================================================
            # RIGHT SHIELD
            # =================================================

            if (
                SHIELDS
                and xMinR is not None
                and xMaxR is not None
                and yMinR is not None
                and yMaxR is not None
            ):


                # -------------------------------------------------
                # RIGHT HAND CENTER
                # -------------------------------------------------

                xc_rh = (
                    xMaxR + xMinR
                ) / 2

                yc_rh = (
                    yMaxR + yMinR
                ) / 2


                xc_rh = int(
                    width * xc_rh
                )

                yc_rh = int(
                    height * yc_rh
                )


                # -------------------------------------------------
                # SHIELD SIZE
                # -------------------------------------------------

                r_width_shield = int(

                    width
                    * (xMaxR - xMinR)
                    / 2
                    * 3.5
                    * scale

                )


                r_height_shield = int(

                    height
                    * (yMaxR - yMinR)
                    / 2
                    * 2
                    * scale

                )


                if (
                    r_width_shield > 0
                    and
                    r_height_shield > 0
                ):


                    # ---------------------------------------------
                    # RESIZE
                    # ---------------------------------------------

                    res3 = cv2.resize(

                        res,

                        (
                            r_width_shield * 2,
                            r_height_shield * 2
                        )

                    )


                    # ---------------------------------------------
                    # INITIAL CROP
                    # ---------------------------------------------

                    start_h = 0
                    start_w = 0

                    stop_h = (
                        r_height_shield * 2
                    )

                    stop_w = (
                        r_width_shield * 2
                    )


                    # ---------------------------------------------
                    # POSITION
                    # ---------------------------------------------

                    f_start_h = (
                        yc_rh -
                        r_height_shield
                    )

                    f_stop_h = (
                        yc_rh +
                        r_height_shield
                    )

                    f_start_w = (
                        xc_rh -
                        r_width_shield
                    )

                    f_stop_w = (
                        xc_rh +
                        r_width_shield
                    )


                    # ---------------------------------------------
                    # TOP
                    # ---------------------------------------------

                    if f_start_h < 0:

                        start_h = -f_start_h

                        f_start_h = 0


                    # ---------------------------------------------
                    # BOTTOM
                    # ---------------------------------------------

                    if f_stop_h > height:

                        stop_h = (
                            res3.shape[0]
                            -
                            (
                                f_stop_h
                                -
                                height
                            )
                        )

                        f_stop_h = height


                    # ---------------------------------------------
                    # LEFT
                    # ---------------------------------------------

                    if f_start_w < 0:

                        start_w = -f_start_w

                        f_start_w = 0


                    # ---------------------------------------------
                    # RIGHT
                    # ---------------------------------------------

                    if f_stop_w > width:

                        stop_w = (
                            res3.shape[1]
                            -
                            (
                                f_stop_w
                                -
                                width
                            )
                        )

                        f_stop_w = width


                    # ---------------------------------------------
                    # CROP
                    # ---------------------------------------------

                    res3 = res3[
                        start_h:stop_h,
                        start_w:stop_w
                    ]


                    # ---------------------------------------------
                    # TARGET
                    # ---------------------------------------------

                    target = frame[
                        f_start_h:f_stop_h,
                        f_start_w:f_stop_w
                    ]


                    # ---------------------------------------------
                    # BLEND
                    # ---------------------------------------------

                    if (

                        target.size > 0

                        and

                        res3.size > 0

                        and

                        target.shape ==
                        res3.shape

                    ):

                        blended = cv2.addWeighted(

                            target,

                            alpha,

                            res3,

                            1.0,

                            1

                        )


                        frame[
                            f_start_h:f_stop_h,
                            f_start_w:f_stop_w
                        ] = blended


            # =================================================
            # GESTURE PREDICTION
            # =================================================

            prediction = None

            pred_prob = 0.0


            # =================================================
            # ONLY PREDICT WHEN BOTH HANDS ARE PRESENT
            # =================================================

            if (

                xMinL is not None

                and

                xMinR is not None

            ):

                try:


                    hand_points = (
                        points_detection_hands(
                            results
                        )
                    )


                    input_data = np.array(
                        [hand_points]
                    )


                    prediction = (
                        model.predict(
                            input_data
                        )[0]
                    )


                    probabilities = (
                        model.predict_proba(
                            input_data
                        )[0]
                    )


                    pred_prob = float(
                        np.max(
                            probabilities
                        )
                    )


                except Exception:

                    prediction = None

                    pred_prob = 0.0


            # =================================================
            # GESTURE CONTROL
            # =================================================

            if (

                prediction is not None

                and

                pred_prob >=
                PREDICTION_THRESHOLD

            ):


                # =================================================
                # SHIELDS OFF
                # =================================================

                if not SHIELDS:


                    # =================================================
                    # KEY 1
                    # =================================================

                    if prediction == "key_1":

                        KEY_1 = True

                        KEY_2 = False
                        KEY_3 = False

                        t1 = datetime.now()

                        print(
                            f"\n🔑 KEY_1 "
                            f"({pred_prob:.2f})"
                        )


                    # =================================================
                    # KEY 2
                    # =================================================

                    elif (

                        prediction == "key_2"

                        and

                        KEY_1

                        and

                        t1 is not None

                    ):


                        t2 = datetime.now()


                        elapsed = (
                            t2 - t1
                        ).total_seconds()


                        if elapsed <= GESTURE_TIMEOUT:

                            KEY_2 = True

                            print(
                                f"\n🔑 KEY_2 "
                                f"({pred_prob:.2f})"
                            )

                        else:

                            KEY_1 = False
                            KEY_2 = False
                            KEY_3 = False

                            t1 = None
                            t2 = None
                            t3 = None

                            print(
                                "\n⚠️ Gesture "
                                "sequence timed out."
                            )


                    # =================================================
                    # KEY 3
                    # =================================================

                    elif (

                        prediction == "key_3"

                        and

                        KEY_1

                        and

                        KEY_2

                        and

                        t2 is not None

                    ):


                        t3 = datetime.now()


                        elapsed = (
                            t3 - t2
                        ).total_seconds()


                        if elapsed <= GESTURE_TIMEOUT:

                            KEY_3 = True

                            SHIELDS = True


                            print("\n")
                            print(
                                "=" * 60
                            )

                            print(
                                "🛡️🛡️🛡️ "
                                "SHIELDS ACTIVATED "
                                "🛡️🛡️🛡️"
                            )

                            print(
                                "=" * 60
                            )

                            print(
                                "Shield remains ON "
                                "until KEY_4."
                            )


                        else:

                            KEY_1 = False
                            KEY_2 = False
                            KEY_3 = False

                            t1 = None
                            t2 = None
                            t3 = None

                            print(
                                "\n⚠️ Gesture "
                                "sequence timed out."
                            )


                # =================================================
                # SHIELDS ON
                # =================================================

                else:


                    # =================================================
                    # KEY 4
                    # =================================================

                    if prediction == "key_4":

                        KEY_1 = False
                        KEY_2 = False
                        KEY_3 = False

                        SHIELDS = False

                        t1 = None
                        t2 = None
                        t3 = None


                        print("\n")
                        print(
                            "=" * 60
                        )

                        print(
                            "🛡️ SHIELDS DEACTIVATED"
                        )

                        print(
                            "=" * 60
                        )


            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            if show_window:

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                # =================================================
                # Q
                # =================================================

                if key == ord("q"):

                    print(
                        "\n🛑 Q pressed."
                    )

                    break


                # =================================================
                # KEYBOARD 1
                # =================================================

                elif key == ord("1"):

                    KEY_1 = True

                    KEY_2 = False
                    KEY_3 = False

                    t1 = datetime.now()

                    print(
                        "\n🔑 KEY_1 "
                        "ACTIVATED BY KEYBOARD"
                    )


                # =================================================
                # KEYBOARD 2
                # =================================================

                elif key == ord("2"):

                    if KEY_1:

                        KEY_2 = True

                        t2 = datetime.now()

                        print(
                            "\n🔑 KEY_2 "
                            "ACTIVATED BY KEYBOARD"
                        )

                    else:

                        print(
                            "\n⚠️ Press 1 first."
                        )


                # =================================================
                # KEYBOARD 3
                # =================================================

                elif key == ord("3"):

                    if KEY_1 and KEY_2:

                        KEY_3 = True

                        SHIELDS = True

                        print("\n")
                        print(
                            "=" * 60
                        )

                        print(
                            "🛡️🛡️🛡️ "
                            "SHIELDS ACTIVATED "
                            "🛡️🛡️🛡️"
                        )

                        print(
                            "=" * 60
                        )

                    else:

                        print(
                            "\n⚠️ Press "
                            "1 then 2 first."
                        )


                # =================================================
                # KEYBOARD 4
                # =================================================

                elif key == ord("4"):

                    KEY_1 = False
                    KEY_2 = False
                    KEY_3 = False

                    SHIELDS = False

                    t1 = None
                    t2 = None
                    t3 = None

                    print("\n")
                    print(
                        "=" * 60
                    )

                    print(
                        "🛡️ SHIELDS "
                        "DEACTIVATED"
                    )

                    print(
                        "=" * 60
                    )


            # =================================================
            # DISPLAY
            # =================================================
            #
            # IMPORTANT:
            #
            # There is NO cv2.putText()
            # for predictions or keys.
            #
            # Therefore:
            #
            # KEY_1
            # KEY_2
            # KEY_3
            # Confidence
            # Prediction
            #
            # WILL NOT APPEAR
            # ON THE CAMERA.
            #
            # =================================================

            if show_window:

                cv2.imshow(
                    "Dr. Strange Shields",
                    frame
                )


            # =================================================
            # VIRTUAL CAMERA
            # =================================================

            if (

                use_virtual_cam

                and

                cam is not None

            ):

                cam.send(
                    frame
                )

                cam.sleep_until_next_frame()


            # =================================================
            # NO WINDOW
            # =================================================

            if not show_window:

                time.sleep(
                    0.01
                )


    # ========================================================
    # KEYBOARD INTERRUPT
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n🛑 Ctrl+C received."
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print("\n")
        print(
            "=" * 60
        )

        print(
            "❌ ERROR DURING EXECUTION"
        )

        print(
            "=" * 60
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        cleanup()