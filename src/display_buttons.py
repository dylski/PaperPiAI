#!/usr/bin/env python3
import gpiod
import gpiodevice
from gpiod.line import Bias, Direction, Edge
import os
import time
import shutil
import subprocess

# ==== CONFIG ====
IMAGE_DIR = "/home/dylski/images/flowers"
DISPLAY_IMAGE = os.path.join(IMAGE_DIR, "output.png")
# Use a timeout (in seconds) for read_edge_events(). 
# A small value (e.g., 1.0) ensures low-CPU usage 
# while keeping the script responsive.
READ_TIMEOUT = 1


DISPLAY_CMD = [
    "/home/dylski/venv/bin/python3",
    "/home/dylski/projects/PaperPiAI/src/display_picture.py",
    "-r",
    "-p",
    DISPLAY_IMAGE
]

# GPIO pins for each button (from top to bottom)
# These will vary depending on platform and the ones
# below should be correct for Raspberry Pi 5.
# Run "gpioinfo" to find out what yours might be.
#
# Raspberry Pi 5 Header pins used by Inky Impression:
#    PIN29, PIN31, PIN36, PIN18.
# These header pins correspond to BCM GPIO numbers:
#    GPIO05, GPIO06, GPIO16, GPIO24.
# These GPIO numbers are what is used below and not the
# header pin numbers.

SW_A = 5
SW_B = 6
SW_C = 16  # Use 16 for 7.3" and use '25' for Impression 13.3"
SW_D = 24

BUTTONS = [SW_A, SW_B, SW_C, SW_D]

# These correspond to buttons A, B, C and D respectively
LABELS = ["A", "B", "C", "D"]

# Create settings for all the input pins, we want them to be inputs
# with a pull-up and a falling edge detection.
INPUT = gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP, edge_detection=Edge.FALLING)

# Find the gpiochip device we need, we'll use
# gpiodevice for this, since it knows the right device
# for its supported platforms.
chip = gpiodevice.find_chip_by_platform()

# Build our config for each pin/line we want to use
OFFSETS = [chip.line_offset_from_id(id) for id in BUTTONS]
line_config = dict.fromkeys(OFFSETS, INPUT)

# Request the lines, *whew*
request = chip.request_lines(consumer="spectra6-buttons", config=line_config)



# ==== FUNCTIONS ====

def get_png_list():
    """Return list of PNG files sorted by mtime (newest first),
       excluding output.png."""
    files = []
    for f in os.listdir(IMAGE_DIR):
        if not f.endswith(".png"):
            continue
        if f == "output.png":
            continue
        fp = os.path.join(IMAGE_DIR, f)
        files.append((fp, os.path.getmtime(fp)))

    # Sort by timestamp newest → oldest
    files.sort(key=lambda x: x[1], reverse=True)

    # Return list of file paths only
    return [f[0] for f in files]


def run_display_renderer():
    """Run the external display rendering python script."""
    print("Running display.py...")
    try:
        subprocess.run(DISPLAY_CMD, check=True)
        print("display.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running display.py: {e}")


def copy_index_to_display(index):
    """Copy the nth newest image to display.png if it exists."""
    images = get_png_list()
    if index < 0 or index >= len(images):
        print(f"No image at index {index}")
        return

    print(f"Copying {images[index]} -> {DISPLAY_IMAGE}")
    shutil.copy2(images[index], DISPLAY_IMAGE)  # copy2 preserves timestamps

    # Run display.py after copying
    run_display_renderer()


def get_current_display_index():
    """
    Determine which index display.png corresponds to,
    by matching timestamps with available images.
    Returns index or None.
    """
    if not os.path.exists(DISPLAY_IMAGE):
        return None

    display_mtime = os.path.getmtime(DISPLAY_IMAGE)
    images = get_png_list()

    for idx, img in enumerate(images):
        if abs(os.path.getmtime(img) - display_mtime) < 0.001:
            return idx
    return None


# ==== BUTTON HANDLERS ====


def press_a():
    print("Button A pressed → latest image")
    copy_index_to_display(0)

def press_b():
    print("Button B pressed → step backwards in time")
    current_index = get_current_display_index()

    if current_index is None:
        print("display.png not matching any known image. Using newest.")
        next_index = 0
    else:
        next_index = current_index + 1

    copy_index_to_display(next_index)

def press_c():
    print("Button C pressed → stepforwards in time")
    current_index = get_current_display_index()

    if current_index is None:
        print("display.png not matching any known image. Using newest.")
        next_index = 0
    else:
        next_index = current_index - 1

    copy_index_to_display(next_index)

def press_d():
    print("Button D pressed: Shutting down")
    try:
        subprocess.run(['sudo', 'shutdown', 'now'], check=False)
    except subprocess.CalledProcessError as e:
        print(f"Error running display.py: {e}")


BUTTON_FN = {
  "A": press_a,
  "B": press_b, 
  "C": press_c, 
  "D": press_d
}


# "handle_button" will be called every time a button is pressed
# It receives one argument: the associated gpiod event object.
def handle_button(event):
    index = OFFSETS.index(event.line_offset)
    gpio_number = BUTTONS[index]
    label = LABELS[index]
    print(f"Button press detected on GPIO #{gpio_number} label: {label}")
    BUTTON_FN[label]()


# ==== MAIN LOOP ====
print(f"Monitoring GPIO lines: {BUTTONS} with gpiod...")
try:
    while True:
        # Crucial Change: read_edge_events() now uses the timeout.
        # This keeps the CPU idle until the kernel signals an event or the timeout is reached.
        for event in request.read_edge_events(READ_TIMEOUT):
            # If an event is received before the timeout, it is processed here
            handle_button(event)
        
        # If no event is received, the loop continues after the timeout, 
        # allowing for clean script termination or other background tasks if needed.
        # This ensures minimal CPU load while waiting.
except KeyboardInterrupt:
    print("\nMonitor stopped by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn unhandled error occurred: {e}")
finally:
    # Release the GPIO lines when exiting
    request.release()
    print("GPIO lines released. Exiting.")

