import argparse
import random
import shutil
import subprocess
import os

def usage():
    print("Usage: python3 generate_picture.py output_dir")
    exit(1)

parser = argparse.ArgumentParser(description="Generate a new random picture.")
parser.add_argument("output_dir", help="Directory to save the output images")
parser.add_argument("--width", default=800, help="The width of the image to generate")
parser.add_argument("--height", default=480, help="The height of the image to generate")
args = parser.parse_args()

# Set the paths
installed_dir = "/home/dylski/Projects/PaperPiAI"
installed_dir = "./"
sd_bin = f"{installed_dir}/OnnxStream/src/build/sd"
sd_model = f"{installed_dir}/models/stable-diffusion-xl-turbo-1.0-anyshape-onnxstream"

output_dir = args.output_dir
shared_file = 'output.png'

steps = 5
seed = random.randint(1, 10000)

# Define flowers and art styles lists
flowers = [
    "single stem rose", "bouquet of tulips", "single sunflower", "single daisy",
    "bouquet of daffodils", "single orchid", "single lily", "bouquet of peonies",
    "wild poppies", "bunch of marigolds", "single iris", "bouquet of chrysanthemums",
    "single hibiscus bloom", "purple violets", "single bluebell stem", "red camellia blossom",
    "single morning glory", "single hydrangea bloom", "single wisteria cluster",
    "single petunia bloom", "yellow narcissus", "bouquet of carnations", "single azalea bloom",
    "single freesia stem", "bunch of snowdrops", "single calla lily", "single lupine flower",
    "single foxglove bloom", "gardenia blossom", "red amaryllis bloom", "bouquet of ranunculus",
    "single gerbera daisy", "single cornflower", "violet gladiolus", "single buttercup",
    "single snapdragon", "single jasmine bloom", "single heather sprig", "bouquet of zinnias",
    "single anthurium bloom", "orange begonia", "bouquet of cosmos flowers", "single sweet pea",
    "single forget-me-not", "single honeysuckle flower", "single magnolia blossom",
    "single posy of wildflowers", "field of sunflowers", "field of tulips", 
    "bouquet of mixed wildflowers", "single posy of daisies"
]

art_styles = [
    "as a cartoon drawing", "as a children's book illustration", "as a vintage movie poster",
    "as a vintage postcard", "using block printing", "in the style of art deco",
    "as a botanical illustration", "as abstract art", "as art nouveau", "in comic style",
    "in cubism style", "as digital pixel art", "as an engraving", "using flat illustration",
    "as folk art", "in geometric art style", "as a midcentury style", "using minimalism style",
    "as a naive style", "in painterly style", "as paper cut art", "in pen and ink",
    "as a pencil illustration", "in pointillism style", "in pop art style", "as primitive cave art",
    "in psychedelic art style", "as scratch art", "as a screenprint", "as sketchy art",
    "as stained glass", "as a vintage retro illustration", "as a woodcut", "using silkscreen",
    "as stencil art", "as a gouache painting", "using collage art", "in manga style",
    "using lithography", "using monoprinting", "as vector art", "in constructivism style",
    "as a charcoal drawing", "in graphic novel style", "as mosaic art", "as a linocut",
    "as a contemporary illustration", "using stencil graffiti"
]


# Select a random subject and art style
subject = random.choice(flowers)
art_style = random.choice(art_styles)

# Create a unique argument for the filename
unique_arg = f"{subject.replace(' ', '_')}_{art_style.replace(' ', '_')}_seed_{seed}_steps_{steps}"
fullpath = os.path.join(output_dir, f"{unique_arg}.png")

# Construct the command
cmd = [
    sd_bin,
    "--xl", "--turbo",
    "--models-path", sd_model,
    "--rpi-lowmem",
    "--prompt", f"{subject} {art_style}",
    "--seed", str(seed),
    "--output", fullpath,
    "--steps", str(steps),
    "--res", f"{args.width}x{args.height}"
]

# Run the command
print(f"Creating image with prompt '{subject} {art_style}'")
print(f"Using seed {seed}")
print(f"Saving to {fullpath}")
print(f"Running command:\n{cmd}")
subprocess.run(cmd)
print("Command executed successfully.")

shared_fullpath = os.path.join(output_dir, shared_file)
shutil.copyfile(fullpath, shared_fullpath)
print(f"Copied to {shared_fullpath}") 

