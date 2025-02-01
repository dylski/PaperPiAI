#@title Jula make and plot
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import shutil
from skimage.measure import shannon_entropy

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
                   colourmap=None, invert=None, bands=0, save_base=None):
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
  if invert is None:
      invert = random.choice([True, False])
  aspect_ratio = height / width

  # Generate the Julia set image
  img = julia_set(c, window_size, width, height, bands, max_iter)
  img = img.astype(float)
  img -= np.min(img)
  img /= np.max(img)
  if invert:
      img = 1 - img

  if colourmap == "RANDOM":
      cmap_names = ['cool', 'twilight', 'magma', 'cividis', 'vanimo', 
              'afmhot', 'hsv', 'YlGn', 'autumn', 'Spectral', 'coolwarm', 
              'inferno', 'seismic', 'managua', 'pink', 'summer', 'YlOrRd', 
              'twilight_shifted', 'winter', 'copperPiYG', 'PuOr', 'plasma', 
              'gist_heat', 'Wistia', 'BrBG', 'berlin', 'hot', 'turbo', 'bwr', 
              'RdYlGn', 'viridis', 'RdGy', 'YlGnBu', 'RdBu', 'RdYlBu', 'spring', 
              'PRGn']
      colourmap = random.choice(cmap_names)

  # Apply colormap
  if colourmap:
    cmap = plt.get_cmap(colourmap)
    img_colored = cmap(img)[:, :, :3]  # Extract RGB channels
    img_colored = (img_colored * 255).astype(np.uint8)
  else:
    img_colored = (img * 255).astype(np.uint8)  # Grayscale

  save_path = None
  if save_base:
      # Create and save the image using PIL
      image = Image.fromarray(img_colored)
      save_path = f"{save_base}/julia_set,c_{c},iter_{max_iter},w_{window_size},cmap_{colourmap},inv_{invert},b_{bands}.png"
      image.save(save_path)
      print("Saved", save_path)

  return img, save_path

# # # Example usage
# c = complex(-0.76, 0.07)
# window_size = 2
# width = 500
# height = 500
# plot_julia_set(c, window_size, width, height, colourmap="RANDOM")
# 
# exit(0)

def tile_images(images):
  tile_w = int(np.ceil(np.sqrt(len(images))))
  tile_h = int(np.ceil(len(images) / tile_w))
  tile_image = np.zeros((tile_h * images[0].shape[0], tile_w * images[0].shape[1]), dtype=np.float32)
  for i, image in enumerate(images):
    image = image.astype(np.float32)
    x = (i % tile_w) * images[0].shape[1]
    y = (i // tile_w) * images[0].shape[0]
    image -= np.min(image)
    image /= np.max(image)
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

#@title Julia GA
import random

def fitness_function(c, window_size, width=100, height=60, bands=0, max_iter=40):
    img = julia_set(c, window_size, width, height, bands, max_iter).astype(np.float32)
    fitness_score = shannon_entropy(img)
    #fitness_score *= metric_contrast(img)

    #fitness_score = metric_fractal_dim(img)
    #fitness_score *= metric_symmetry(img)
    return fitness_score, img

def initialize_population(pop_size, max_iter):
    """Initialize a population with random complex numbers and window sizes."""
    population = []
    for _ in range(pop_size):
        real = random.uniform(-2, 2)
        imag = random.uniform(-2, 2)
        complex_num = complex(real, imag)
        window_size = random.uniform(1.0, 2.0)  # Start with a random window size
        iterations = random.randint(40, max_iter)  # Random number of iterations
        bands = random.randint(5, 23)  # Random number of iterations
        population.append((complex_num, window_size, iterations, bands))
    return population

def crossover(parent1, parent2):
    """Create a child by mixing the 'c' values and window sizes of two parents."""
    c_real = (parent1[0].real + parent2[0].real) / 2
    c_imag = (parent1[0].imag + parent2[0].imag) / 2
    c = complex(c_real, c_imag)

    window_size = random.uniform(min(parent1[1], parent2[1]), max(parent1[1], parent2[1]))
    iterations = random.randint(min(parent1[2], parent2[2]), max(parent1[2], parent2[2]))
    bands = random.randint(min(parent1[3], parent2[3]), max(parent1[3], parent2[3]))

    return (c, window_size, iterations, bands)

def mutate(individual):
    """Mutate an individual by randomly changing the complex constant 'c' and window size."""
    mutation_rate = 4
    mutation_type = random.choice(['complex', 'window', 'iterations', 'bands'])
    if mutation_type == 'complex':
        # Mutate the complex constant 'c' by a small random change
        individual = (individual[0] + random.uniform(-0.55, 0.55) * mutation_rate 
                      + 1j * random.uniform(-0.55, 0.55) * mutation_rate,
                      individual[1],
                      individual[2],
                      individual[3],
                      )
    elif mutation_type == 'window':
        # Mutate the window size slightly
        individual = (individual[0],
                      individual[1] + random.uniform(-0.1, 0.1) * mutation_rate,
                      individual[2],
                      individual[3],
                      )
    elif mutation_type == 'iterations':
        # Mutate the number of iterations
        individual = (individual[0],
                      individual[1],
                      max(40, individual[2] + random.randint(-10, 10) * mutation_rate),
                      individual[3],
                      )
    elif mutation_type == 'bands':
        # Mutate the number of bands
        individual = (individual[0],
                      individual[1],
                      individual[2],
                      min(23, max(5, individual[3] + random.randint(-10, 10) * mutation_rate)),
                      )
    else:
        raise ValueError("Invalid mutation type")

    return individual

def select(population, fitness_values, num_parents):
    """Select the best individuals based on fitness."""

    fitness_values += np.random.uniform(0, 0.00001, size=len(fitness_values))
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    return sorted_population[:num_parents]

def genetic_algorithm(pop_size, generations, max_iter, save_tiled_base=None, plot=False):
    if plot:
        import display_manager as dm
        display = dm.DisplayManager(num_images=2, num_plots=1)
    else:
        display = None
        
    fitness_history = []
    population = initialize_population(pop_size, max_iter)
    fitness_imgs = [fitness_function(ind[0], ind[1], max_iter=ind[2], bands=ind[3]) for ind in population]
    fitness_values, images = zip(*fitness_imgs)
    tiled = (tile_images(images) * 255).astype(np.uint8)
    if save_tiled_base:
        _img = Image.fromarray(tiled, "L")
        _img.save(f"{save_tiled_base}_gen{0:03}.png")
    if display:
        display.update_image(0, tiled)
    print(f"Generation 0: Best fitness = {max(fitness_values)}")
    c, window_size, iteration, bands = population[fitness_values.index(max(fitness_values))]
    print(f"c={c}, window_size={window_size}, iteration={iteration}, bands={bands}")
    best_image, _ = plot_julia_set(c, window_size, bands=bands, width=100, height=60,
                                colourmap="twilight", max_iter=iteration)
    if display:
        display.update_image(1, best_image)
    fitness_history.append(fitness_values)
    #plot_history(fitness_history)
    if display:
        display.update_scatter(0, fitness_history)

    for generation in range(generations):
        parents = select(population, fitness_values, pop_size // 2)

        # Crossover
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child = crossover(parent1, parent2)
            next_generation.append(child)

        # Mutation
        next_generation = [mutate(child) for child in next_generation]
        population = parents + next_generation
        fitness_imgs = [fitness_function(ind[0], ind[1], max_iter=ind[2]) for ind in population]
        fitness_values, images = zip(*fitness_imgs)

        tiled = tile_images(images)
        if display:
            display.update_image(0, tiled)
        tiled = (tiled * 255).astype(np.uint8)
        #_img = Image.fromarray(tiled, "L")
        #_img.save(f"{save_base}_gen{generation+1:03}.png")
        print(f"Generation {generation + 1}: Best fitness = {max(fitness_values)}")
        c, window_size, iteration, bands = population[fitness_values.index(max(fitness_values))]
        print(f"c={c}, window_size={window_size}, iteration={iteration}, bands={bands}")
        best_image, _ = plot_julia_set(c, window_size, width=100, height=60,
                                    colourmap="twilight", bands=bands, max_iter=iteration)
        fitness_history.append(fitness_values)
        if display:
            display.update_image(1, best_image)
            display.update_scatter(0, fitness_history)


    # Return the best individual after all generations
    best_individual = population[fitness_values.index(max(fitness_values))]
    return best_individual

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save_dir", default="./",
        help="Save path for iamges")
    ap.add_argument("-g", "--generations",
                    default=10, help="Number of generations")
    ap.add_argument("-p", "--population",
                    default=10, help="Size of population")
    ap.add_argument("-t", "--tiled", action="store_true",
                    default=False, help="Save tiled population images")
    ap.add_argument("--width", default=800, help="The width of the display")
    ap.add_argument("--height", default=480, help="The height of the display")
    args = vars(ap.parse_args())

    assert int(args["population"]) % 4 == 0, "Population must be multiple of 4"
    # Run the genetic algorithm
    best_individual = genetic_algorithm(
            pop_size=int(args["population"]), generations=int(args["generations"]),
            max_iter=140, save_tiled_base=None)

    # Plot the best Julia set found, scaled to 800x480
    c, window_size, iteration, bands = best_individual
    print(f"c={c}, window_size={window_size}, iteration={iteration}, bands={bands}")
    _, save_path = plot_julia_set(c, window_size, width=800, height=480, bands=bands, max_iter=iteration,
                   colourmap="RANDOM", save_base=args["save_dir"])

    shared_fullpath = os.path.join(args["save_dir"], "output.png")
    shutil.copyfile(save_path, shared_fullpath)
    print(f"Copied to {shared_fullpath}") 
