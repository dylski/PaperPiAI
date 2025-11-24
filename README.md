# Updates

* 24th Nov 2025: Inky 13.3" support with latest Trixie OS
* 1st Jan 2025: Added support for OnnxStream's new custom resolutions and updated some documentation.
Special thanks to [Vito Plantamura](https://github.com/vitoplantamura), [Delph](https://github.com/Delph) and [Roger](https://github.com/g7ruh)

# PaperPiAI - Raspberry Pi Zero Generative Art E-Paper Frame

PaperPiAI is a standalone Raspberry Pi Zero 2 powered e-ink picture frame
running stable diffusion generating an infinite array of pictures.

This default set-up generates random flower pictures with random styles that
I've found to work particularly well on the Inky Impressions 7.3" 7-colour
e-ink display, i.e. low-colour palette styles and simple designs.

![Blooming Great!](https://raw.githubusercontent.com/dylski/PaperPiAI/refs/heads/main/assets/paperpiai_examples.jpg)

Once set up the picture frame is fully self-sufficient, able to generate unique
images with no internet access until the end of time (or a hardware failure -
which ever comes first).

Each image takes about 30 minutes to generate and about 30 seconds to refresh
to the screen.  You can change the list of image subjects and styles in the
prompt file or provide your own. Ideally I'd like to generate the image
prompts with a local LLM but have not found one that runs on the RPi Zero 2
yet. It would not have to run fast - just fast enough to generate a new prompt
within 23 hours to have a new picure every day.

OnnxStream now supports custom resolutions for Stable Diffusion XL Turbo 1.0
so we can render directly for the display size. If for some reason you cannot
make use of this feature (e.g. using a different model) there is support
to use 'intelligent' cropping that uses [salient spectral feature
analysis](https://towardsdatascience.com/opencv-static-saliency-detection-in-a-nutshell-404d4c58fee4)
to guide the crop (landscape or portrait) towards most interesting part of
the image. This was needed in an earlier version when we could only generate
512x512 images.

# Install

* **Raspberry Pi Zero 2**
* **Inky Impression 7.3"** 7-colour e-ink display. For Waveshare displays, see
  [here](https://github.com/dylski/PaperPiAI/issues/12#issuecomment-2603268581)
* Picture frame, ideally with deep frame to accommodate RPi Zero
* Heatsink (optional) - I saw a max of 70°C (ambient was ~21°C) but one might
  be useful in a hot area or confined space
* **Raspbian Bullseye Lite**. A similar set-up ought to work with Bookwork
  (install inky 2.0 using Pimoroni's instructions) but I had odd slowdowns
using Bookwork which I could not resolve.

##  Increase swapfile size for compilation

Edit **/etc/dphys-swapfile** (e.g. `sudo vim /etc/dphys-swapfile`) and change
the value of **CONF_SWAPSIZE** to 1024. You _might_ be able to get away with a
smaller swap size but it's been reported that the build process stalls with a
swap size of 256.

Then restart swap with `sudo /etc/init.d/dphys-swapfile restart`

## Enable E-paper interfaces

run `sudo raspi-config` and enable **SPI interface** and **I2C interface**

## Install required components

Firstly download this repository somewhere with:

```
sudo apt install git
git clone https://github.com/dylski/PaperPiAI.git
```
Then run the installation script:
```
cd PaperPiAI
scripts/install.sh
```
`scripts/install.sh` has all the commands needed to install all the required
system packages, python libraries and [OnnxStream](https://github.com/vitoplantamura/OnnxStream)
 - Stable Diffusion for the Raspberry Pi Zero.  If you are feeling brave then
run `install.sh` in the directory you want to install everything, otherwise run
each command manually.

The whole process takes a _long_ time, i.e. several hours. If you are building
in a RPi 4 or 5 you can speed it up by appending ` -- -j4`  or ` -- -all` to
the `cmake --build . --config Release` lines in `install.sh`. This instructs
the compiler to use four cores or all cores, respectively. This speed up does
not work on the RPi Zero 2 as it only has 512MB RAM. Also note that 8GB of
model parameters will be downloaded. Depending on your wifi signal and braodband
speed this can also take a long time. It is recommended to position the RPi such
that the e-ink display does not impair the wifi signal!

Apologies for the rather manual approach - my install-fu is not up to scratch!

# Generating and displaying

## Generating

You need to run `generate_picture.py` with the resolution of the display and a
target directory to save the images. The Inky Impressions has a resolution of
800x480 so for a landscape image the command would be:

`python src/generate_picture.py --width=800 --height=480 output_dir`

This generates a new image with a unique name based on the prompt, and a copy
called 'output.png' to make it simple to display.

Note that if you install the python packages into a virtual env (as the script
above does) then you need to use that python instance, e.g.:

`venv/bin/python src/generate_picture.py --width=800
--height=480 /tmp`

For more options, run the `generate_picture.py` script with the `-h` or `--help`
flags to see full usage.

## Displaying

To send to the display use `venv/bin/python src/display_picture.py -r <image_name>`

Tie `-r` option skips any intelligent cropping (as this is no longer needed)
and just resizes the image to make sure it fits the display.

## Portrait display

To generate portrait images to display on portrait-oriented display switch the
width and height values for `generate_picture.py` and include the `-p` with the
display_picture.py script.  I.e. for the Inky Impression:

`venv/bin/python src/generate_picture.py --width=480 --height=800 output_dir`

and 

`venv/bin/python src/display_picture.py -r -p output_dir/output.png`


For more options, run the `display_picture.py` script with the `-h` or `--help`
flags to see full usage.

## Automating

To automate and generate an image per day created an executable script called `cron_flower` that runs the generation and display code. My version contains the following lines:

```#!/bin/bash
cd "/home/dylski"
./venv/bin/python PaperPiAI/src/generate_picture.py --width 800 --height 480 images
./venv/bin/python PaperPiAI/src/display_picture.py -r images/output.png
```
Obviously change yours to point to where your code is.

Then I added the entry in crontab (run `crontab -e` to edit your crontab file):
`0 0 * * * /home/dylski/bin/cron_flower`
to run `cron_flower` every day at midnight.

Note that e-paper displays are
suspectible to temperature. Depending on your Pi Zero's environment, it 
may get hot for an extended period of
time which could cause the display to render with some discoloration. This can be
avoided by delaying the display update after generating the image.

## Prompts
The `prompts` directory stores JSON files which can be used to provide promps.
A prompt file is simply just an array of array of prompt fragments, one option
is chosen from each array at random and it is concatenated together. Custom
prompts can be provided with the `--prompt` option, or the file to use can be
specified with the `--prompts` option.

# Storage

All the generated images are currently retained locally. Each image is ~1.2MB
(assuming 800x480px resolution), so generating an image every 24 hours for
2.25 years would take up ~1GB storage. If you are generating images at a faster
rate and/or storage is limited then this would lead to issues.

A simple fix is to not save images with unique names, i.e. change [this line](
https://github.com/dylski/PaperPiAI/blob/main/src/generate_picture.py#L38) from

`fullpath = os.path.join(output_dir, f"{unique_arg}.png")`

to

`fullpath = os.path.join(output_dir, shared_file)`

and comment out lines 61 - 63.

# Inky 13.3" Update

To use PaperPiAI on an Inky 13.3" screen with Trixie (Debian 13) takes a little exta wrangling as RAM is tight. Here are the instructions for maximising free memory which worked for me. 

Steps 1-3 are probably the most critical for memory. Step 5 important to install latest inky module (the script in the repo forces an older 1.50 version needed for the 7.3" screens).

## 1\. Disable ZRAM-based swap and Maximize Swap File Size

Trixie defaults to **ZRAM** (compressed RAM swap), which uses precious physical memory. We must disable it and enforce a large $\mathbf{2\text{GB}}$ swap file on the SD card using `rpi-swap`.

1.  **Stop the default ZRAM service:**
    ```bash
    sudo systemctl stop dev-zram0.swap
    ```
2.  **Use `rpi-swap` to create a 2GB persistent swapfile:**
      * Ensure the `rpi-swap` package is installed: `sudo apt install rpi-swap`
      * Create the configuration file to override the ZRAM default:
        ```bash
        sudo nano /etc/rpi/swap.conf.d/80-use-swapfile.conf
        ```
      * Add the following content to set the Swapfile mechanism and a fixed size of **2048MiB (2GB)**:
        ```ini
        [Main]
        Mechanism=swapfile
        [File]
        FixedSizeMiB=2048
        ```
      * **Reboot** the Pi immediately for this change to take effect: `sudo reboot`
3.  **Verify the change** after rebooting by checking the swap size:
    ```bash
    free -m
    ```

-----

## 2\. Lower Kernel Memory Reservation (`vm.min_free_kbytes`)

This parameter reserves a block of RAM for the kernel. By default, it's too large ($\mathbf{16\text{MB}}$) for a 512MB system and effectively triggers an OOM killer.

1.  Edit the system configuration file:
    ```bash
    sudo nano /etc/sysctl.d/98-rpi.conf
    ```
2.  Change the line `vm.min_free_kbytes = 16384` to the minimum safe limit of **1024** (1MB):
    ```
    vm.min_free_kbytes = 1024
    ```

-----

## 3\. Aggressively Use Swap (`vm.swappiness`)

This tells the kernel to aggressively use the new 2GB swap file to push idle data out of physical RAM, keeping it clear for the demanding Stable Diffusion job.

1.  In the same configuration file (`98-rpi.conf`), add this setting:
    ```
    vm.swappiness = 90
    ```
2.  Apply the changes immediately:
    ```bash
    sudo sysctl --load /etc/sysctl.d/98-rpi.conf
    ```

-----

## 4\. Disable Unnecessary System Services (Optional?)

Disable background services that consume memory but are not required for a headless e-ink frame. You can run `systemctl list-units --type=servive --state=running` to get a list of running services. I don't know how much these helped but I disabled modem, avah1 (makes the rpi discoverable on the intranet - I'm logging in locally or with IP address) and bluetooth.

```bash
# Stop and disable Mobile Broadband Manager
sudo systemctl disable ModemManager.service
sudo systemctl stop ModemManager.service

# Stop and disable Zeroconf/mDNS service discovery
sudo systemctl disable avahi-daemon.service
sudo systemctl stop avahi-daemon.service

# Stop and disable bluetooth Daemon
sudo systemctl disable bluetooth.service
sudo systemctl stop bluetooth.service
```


## 5\. Update Inky Library (For Inky 13.3")
If you are using a Pimoroni Inky display (or run into driver issues), ensure the inky package is the latest version, as older versions may have compatibility issues with newer OS kernels or Python versions.

Upgrade command:

```Bash

# Assuming you are in your virtual environment
python -m pip install inky[rpi] --upgrade
```
B. Display Resolution in Configuration
The default scripts often target smaller displays. You must specify the correct resolution 1600x1200 for your 13.3" display when running the generation script.

Example for 1600x1200 resolution (Landscape):

```Bash

# Generate the image
venv/bin/python src/generate_picture.py --width=1600 --height=1200 output_dir

# Display the image
venv/bin/python src/display_picture.py output_dir/output.png
```

For Portrait orientation, swap the width/height and add the -p flag:

```Bash

# Generate the image
venv/bin/python src/generate_picture.py --width=1200 --height=1600 output_dir

# Display the image (note the -p flag)
venv/bin/python src/display_picture.py -p output_dir/output.png
```
