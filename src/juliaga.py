#@title Jula make and plot
import argparse
import clip
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import shutil
from skimage.measure import shannon_entropy
import torch

cmap_names = ['cool', 'twilight', 'magma', 'cividis', 'vanimo', 
      'afmhot', 'hsv', 'YlGn', 'autumn', 'Spectral', 'coolwarm', 
      'inferno', 'seismic', 'managua', 'pink', 'summer', 'YlOrRd', 
      'twilight_shifted', 'winter', 'copper', 'PiYG', 'PuOr', 'plasma', 
      'gist_heat', 'Wistia', 'BrBG', 'berlin', 'hot', 'turbo', 'bwr', 
      'RdYlGn', 'viridis', 'RdGy', 'YlGnBu', 'RdBu', 'RdYlBu', 'spring', 
      'PRGn']

def julia_set(c, window_size, width, height, bands, max_iter=40):
    """Generate a Julia set image with a dynamic window around the complex number c."""
    # Define the real and imaginary ranges for the window around c
    aspect_ratio = height / width
    window_width = window_size
    window_height = window_size * aspect_ratio  # Target render aspect ratio

    x_min = c.real - window_width / 2
    x_max = c.real + window_width / 2
    y_min = c.imag - window_height / 2
    y_max = c.imag + window_height / 2

    # Create meshgrid for the complex plane
    x, y = np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    img = np.zeros(Z.shape, dtype=int)

    # Iteration to generate the Julia set
    for i in range(height):
        for j in range(width):
            z = Z[i, j]
            iter_count = 0
            while abs(z) <= 2 and iter_count < max_iter:
                z = z**2 + c
                iter_count += 1
            img[i, j] = iter_count
    if bands:
        img = img % bands
    return img


def plot_julia_set(c, window_size, width, height, max_iter=40,
                   colourmap=None, invert=False, bands=0, save_base=None):
  """
  Generates and saves a Julia set image.

  Args:
    c: Complex constant for the Julia set.
    window_size: Size of the window to explore.
    width: Width of the image in pixels.
    height: Height of the image in pixels.
    bands: Turn into banded data
    max_iter: Maximum number of iterations (default: 40).
    colourmap: Matplotlib colormap to use (default: "twilight_shifted").
    invert: Invert the colormap (default: False).
    save_path: Path to save the image (default: "julia_set").

  Returns:
    None.
  """
  aspect_ratio = height / width

  # Generate the Julia set image
  img = julia_set(c, window_size, width, height, bands, max_iter)
  img = img.astype(float)
  img -= np.min(img)
  max_val = np.max(img)
  if max_val:
    img /= max_val 
  if invert:
      img = 1 - img

  assert colourmap is not None and colourmap != "RANDOM"

  if colourmap:
    cmap = plt.get_cmap(colourmap)
    img_coloured = cmap(img)[:, :, :3]  # Extract RGB channels
    img_coloured = (img_coloured * 255).astype(np.uint8)
  else:
    img_coloured = (img * 255).astype(np.uint8)  # Grayscale

  save_path = None
  if save_base:
      # Create and save the image using PIL
      image = Image.fromarray(img_coloured)
      save_path = f"{save_base}/julia_set,c_{c},iter_{max_iter},w_{window_size},cmap_{colourmap},inv_{invert},b_{bands}.png"
      image.save(save_path)
      print("Saved", save_path)

  return img_coloured, save_path

def tile_images(images):
  tile_w = int(np.ceil(np.sqrt(len(images))))
  tile_h = int(np.ceil(len(images) / tile_w))
  if images[0].ndim == 2:
    tile_image = np.zeros(
            (tile_h * images[0].shape[0], 
             tile_w * images[0].shape[1]), dtype=np.float32)
  else:
    tile_image = np.zeros(
            (tile_h * images[0].shape[0],
             tile_w * images[0].shape[1], 
             3), dtype=np.float32)
  for i, image in enumerate(images):
    image = image.astype(np.float32)
    x = (i % tile_w) * images[0].shape[1]
    y = (i // tile_w) * images[0].shape[0]
    image -= np.min(image)
    max_val = np.max(image)
    if max_val:
      image /= max_val 
    tile_image[y:y+images[0].shape[0], x:x+images[0].shape[1]] = image
  return tile_image

def metric_shannon(grey_image):
    return shannon_entropy(grey_image)  # Higher entropy is better

def metric_contrast(grey_image):
    return np.std(grey_image)

def metric_symmetry(grey_image):
    h_flip = np.flip(grey_image, axis=1)
    v_flip = np.flip(grey_image, axis=0)
    h_diff = np.mean(np.abs(grey_image - h_flip))
    v_diff = np.mean(np.abs(grey_image - v_flip))
    symmetry = 1 / (h_diff + v_diff + 1e-6)
    return symmetry

def fitness_function(c, window_size, bands, max_iter, cmap=None, width=100, height=60):
    # For grey, using image processing metrics
    #img = julia_set(c, window_size, width, height, bands, max_iter).astype(np.float32)
    img, _ = plot_julia_set(
            c, window_size, width=100, height=60,
            bands=bands, max_iter=max_iter, colourmap=cmap)  #, save_base="/tmp")
    grey = img.mean(axis=2)
    fitness_score = shannon_entropy(grey)
    #fitness_score *= metric_contrast(img)
    #fitness_score = metric_fractal_dim(img)
    #fitness_score *= metric_symmetry(img)
    return fitness_score, img

"""
CLIP compare on 
    Option 1 (Simplest): "A beautiful and aesthetic image."
    Option 2 (Slightly More Detail): "Beautiful, artistic, and inspiring imagery."
    Option 3 (Focusing on Visuals): "Vibrant, elegant, and visually stunning."
    Option 4 (Adding Style Hints): "Classic and modern aesthetic design."
    Option 5 (Evoking Feelings): "Joyful and inspiring aesthetic art."
    Option 6 (A touch more comprehensive): "Beautiful art, design, and nature."
"""

def fitness_function_clip(c, window_size, bands, max_iter, cmap):
    img, _ = plot_julia_set(
            c, window_size, width=224, height=134,
            bands=bands, max_iter=max_iter, colourmap=cmap)
    prompt = "Beautiful, artistic, detailed, stunning."
    fitness_score = clip_match(img, prompt)
    return fitness_score, img

text_features = None
def clip_match(image, prompt):
    global text_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)
    
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0).to(device)
    if text_features is None:
        text = clip.tokenize([prompt]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        if text_features is None:
            text_features = model.encode_text(text)
    
    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()

def initialize_population(pop_size, max_iter):
    """Initialize a population with random complex numbers and window sizes."""
    population = []
    for _ in range(pop_size):
        real = random.uniform(-2, 2)
        imag = random.uniform(-2, 2)
        complex_num = complex(real, imag)
        window_size = random.uniform(1.0, 2.0)  # Start with a random window size
        iterations = random.randint(40, max_iter)  # Random number of iterations
        bands = random.randint(5, 140)  # Random number of iterations
        cmap = random.choice(cmap_names)
        population.append((complex_num, window_size, iterations, bands, cmap))
    return population

def crossover(parent1, parent2):
    """Create a child by mixing the 'c' values and window sizes of two parents."""
    crossover_prob = 0.1

    if np.random.uniform(0, 1) < crossover_prob:
        c_real = (parent1[0].real + parent2[0].real) / 2
        c_imag = (parent1[0].imag + parent2[0].imag) / 2
        c = complex(c_real, c_imag)
    else:
        c = random.choice([parent1[0], parent2[0]])

    if np.random.uniform(0, 1) < crossover_prob:
        window_size = random.uniform(min(parent1[1], parent2[1]), max(parent1[1], parent2[1]))
    else:
        window_size = random.choice([parent1[1], parent2[1]])

    if np.random.uniform(0, 1) < crossover_prob:
        iterations = random.randint(min(parent1[2], parent2[2]), max(parent1[2], parent2[2]))
    else:
        iterations = random.choice([parent1[2], parent2[2]])

    if np.random.uniform(0, 1) < crossover_prob:
        bands = random.randint(min(parent1[3], parent2[3]), max(parent1[3], parent2[3]))
    else:
        bands = random.choice([parent1[3], parent2[3]])

    if np.random.uniform(0, 1) < crossover_prob:
        cmap = random.choice([parent1[4], parent2[4]])
    else:
        cmap  = random.choice([parent1[4], parent2[4]])

    return (c, window_size, iterations, bands, cmap)

def mutate(individual):
    """Mutate an individual by randomly changing the complex constant 'c' and window size."""
    mutation_rate = 4
    mutation_type = random.choice(['complex', 'window', 'iterations', 'bands', 'cmap'])

    # Mutate the number of bands
    individual = [individual[0],
                  individual[1],
                  individual[2],
                  individual[3],
                  individual[4],
                  ]
    if mutation_type == 'complex':
        # Mutate the complex constant 'c' by a small random change
        #individual[0] += (random.uniform(-0.55, 0.55) * mutation_rate 
        #                  + 1j * random.uniform(-0.55, 0.55) * mutation_rate)

        individual[0] = individual[0] + random.uniform(-0.55, 0.55) * mutation_rate + 1j * random.uniform(-0.55, 0.55) * mutation_rate

    elif mutation_type == 'window':
        individual[1] += random.uniform(-0.1, 0.1) * mutation_rate

    elif mutation_type == 'iterations':
        individual[2] = max(
            40, individual[2] + random.randint(-10, 10) * mutation_rate)

    elif mutation_type == 'bands':
        individual[3] = min(
            140, max(5, individual[3] + random.randint(-10, 10) * mutation_rate))

    elif mutation_type == 'cmap':
        individual[4] = random.choice(cmap_names)
    else:
        raise ValueError("Invalid mutation type")

    return individual

def select(population, fitness_values, num_parents):
    """Select the best individuals based on fitness."""

    fitness_values += np.random.uniform(0, 0.00001, size=len(fitness_values))
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    return sorted_population[:num_parents]

def select_tournaments(population, fitness_values, num_parents):
    """Select the individuals based on fitness tournaments. A tournament for each parent."""

    tournament_size = len(population) // num_parents
    remainder = len(population) % num_parents
    fitness_values += np.random.uniform(0, 0.00001, size=len(fitness_values))  # Break ties

    # Shuffle *indices*, not the population itself.
    population_indices = list(range(len(population)))
    random.shuffle(population_indices)

    winners = []
    start_i = 0
    for t in range(num_parents):
        end_i = start_i + tournament_size + (1 if t < remainder else 0)

        # Get the contestants using the shuffled indices
        contestant_indices = population_indices[start_i:end_i]
        contestants = [population[i] for i in contestant_indices]
        contestant_fitnesses = [fitness_values[i] for i in contestant_indices]

        if contestants: # Check if there are any contestants (important for edge cases).
            winner_index = contestant_indices[np.argmax(contestant_fitnesses)]  # Index of the winner in original population
            winner = population[winner_index] # Get the winner from the original population
            winners.append(winner)

        start_i = end_i

    return winners

CLIP = True  # False  # True

def genetic_algorithm(pop_size, generations, max_iter, save_tiled_base=None, plot=False):
    if plot:
        import display_manager as dm
        display = dm.DisplayManager(num_images=2, num_plots=1)
    else:
        display = None
        
    fitness_history = []
    population = initialize_population(pop_size, max_iter)
    if CLIP:
        fitness_imgs = [fitness_function_clip(ind[0], ind[1], max_iter=ind[2], bands=ind[3], cmap=ind[4]) for ind in population]
    else:
        fitness_imgs = [fitness_function(ind[0], ind[1], max_iter=ind[2], bands=ind[3], cmap=ind[4]) for ind in population]
    fitness_values, images = zip(*fitness_imgs)
    tiled = (tile_images(images) * 255).astype(np.uint8)
    if save_tiled_base:
        _img = Image.fromarray(tiled, "L")
        _img.save(f"{save_tiled_base}_gen{0:03}.png")
    if display:
        display.update_image(0, tiled)
    print(f"Generation 0: Best fitness = {max(fitness_values)}")
    c, window_size, iteration, bands, cmap = population[fitness_values.index(max(fitness_values))]
    print(f"c={c}, window_size={window_size}, iteration={iteration}, bands={bands}, cmap={cmap}")
    best_image, _ = plot_julia_set(c, window_size, bands=bands, width=100, height=60,
                                colourmap=cmap, max_iter=iteration)
    if display:
        display.update_image(1, best_image)
    fitness_history.append(fitness_values)
    np.save("/tmp/fitness_history", fitness_history)

    #plot_history(fitness_history)
    if display:
        display.update_scatter(0, fitness_history)

    for generation in range(generations):
        parents = select_tournaments(population, fitness_values, pop_size // 2)
        #parents = select(population, fitness_values, pop_size // 2)

        # Crossover
        next_generation = []
        for i in range(0, len(parents)):
            i_neighbour = (i + 1) % len(parents)
            parent1, parent2 = parents[i], parents[i_neighbour]
            child = crossover(parent1, parent2)
            next_generation.append(child)

        # Mutation
        next_generation = [mutate(child) for child in next_generation]
        population = parents + next_generation
        if CLIP:
            fitness_imgs = [fitness_function_clip(ind[0], ind[1], max_iter=ind[2], bands=ind[3], cmap=ind[4]) for ind in population]
        else:
            fitness_imgs = [fitness_function(ind[0], ind[1], max_iter=ind[2], bands=ind[3], cmap=ind[4]) for ind in population]
        fitness_values, images = zip(*fitness_imgs)

        tiled = tile_images(images)
        if display:
            display.update_image(0, tiled)
        tiled = (tiled * 255).astype(np.uint8)
        #_img = Image.fromarray(tiled, "L")
        #_img.save(f"{save_base}_gen{generation+1:03}.png")
        print(f"Generation {generation + 1}: Best fitness = {max(fitness_values)}")
        c, window_size, iteration, bands, cmap = population[fitness_values.index(max(fitness_values))]
        print(f"c={c}, window_size={window_size}, iteration={iteration}, bands={bands}, cmap={cmap}")
        best_image, _ = plot_julia_set(c, window_size, width=100, height=60,
                                    colourmap=cmap, bands=bands, max_iter=iteration)
        fitness_history.append(fitness_values)
        np.save("fitness_history", fitness_history)
        if display:
            display.update_image(1, best_image)
            display.update_scatter(0, fitness_history)


    # Return the best individual after all generations
    best_individual = population[fitness_values.index(max(fitness_values))]
    return best_individual

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save_dir", default="",
        help="Save path for iamges")
    ap.add_argument("-g", "--generations",
                    default=3, help="Number of generations")
    ap.add_argument("-p", "--population",
                    default=12, help="Size of population")
    ap.add_argument("-t", "--tiled", action="store_true",
                    default=False, help="Save tiled population images")
    ap.add_argument("-d", "--display", action="store_true",
                    default=False, help="Display GA populations")
    ap.add_argument("--width", default=800, help="The width of the display")
    ap.add_argument("--height", default=480, help="The height of the display")
    args = vars(ap.parse_args())

    assert int(args["population"]) % 4 == 0, "Population must be multiple of 4"
    # Run the genetic algorithm
    best_individual = genetic_algorithm(
            pop_size=int(args["population"]), generations=int(args["generations"]),
            max_iter=140, save_tiled_base=None, plot=args["display"])

    # Plot the best Julia set found, scaled to 800x480
    c, window_size, iteration, bands, cmap = best_individual
    print(f"c={c}, window_size={window_size}, iteration={iteration}, bands={bands}, cmap={cmap}")
    _, save_path = plot_julia_set(
            c, window_size, width=800, height=480, bands=bands, max_iter=iteration,
            colourmap=cmap, save_base=args["save_dir"])

    if args["save_dir"]:
        shared_fullpath = os.path.join(args["save_dir"], "output.png")
        shutil.copyfile(save_path, shared_fullpath)
        print(f"Copied to {shared_fullpath}") 
