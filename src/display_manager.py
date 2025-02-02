import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class DisplayManager:
    def __init__(self, num_images=0, num_plots=1, figsize=None):
        if figsize is None:
            figsize = (5 * (num_images + num_plots), 5)

        self.fig, axes = plt.subplots(1, num_images + num_plots, figsize=figsize)
        self.axes = [axes] if num_images + num_plots == 1 else axes

        self.scatter_points = {}
        self.mean_points = {}

        self.num_images = num_images

        for i, ax in enumerate(self.axes):
            if i < num_images:
                # Initialize image display, handling both grayscale and RGB
                initial_image = np.random.uniform(size=(100, 100, 3)).astype(float) # Default to RGB for initialization
                self.image_displays = [] # List to store image display objects

                display = ax.imshow(initial_image) # No cmap for RGB
                self.image_displays.append(display) # Store the image display object
                # No colorbar for RGB

            else:
                ax.set_title('Fitness History')
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness')  # Corrected y-axis label
                ax.grid(True)

                plot_index = i - num_images
                self.scatter_points[plot_index] = []
                self.mean_points[plot_index] = []

        plt.tight_layout()
        plt.show(block=False)

    def update_image(self, index, new_image):
        """Update image at given index, handling grayscale and RGB."""
        if new_image.ndim == 2:  # Grayscale image
            #if self.axes[index].images[0].get_cmap() is None: # Check if it's RGB
            #    self.axes[index].images[0].set_cmap("twilight_shifted") # Set cmap if it's grayscale
            self.axes[index].images[0].set_data(new_image)
        elif new_image.ndim == 3:  # RGB image
            #if self.axes[index].images[0].get_cmap() is not None: # Check if it's grayscale
            #    self.axes[index].images[0].set_cmap(None) # Remove cmap if it's RGB
            self.axes[index].images[0].set_data(new_image)
        else:
            raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

        self.axes[index].images[0].set_clim(
                vmin=new_image.min(), vmax=new_image.max()) # Ensure correct scaling

        self.axes[index].relim() # Recalculate limits
        self.axes[index].autoscale_view() # Autoscale the view
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_scatter(self, plot_index, generation_data):
        """
        Update scatter plot with new generation data
        
        Args:
            plot_index: Index of the plot to update
            generation_data: List of fitness values for current generation
        """
        ax = self.axes[plot_index + self.num_images]
        
        # Clear previous points if any
        if self.scatter_points[plot_index]:
            for scatter in self.scatter_points[plot_index]:
                scatter.remove()
        if self.mean_points[plot_index]:
            for scatter in self.mean_points[plot_index]:
                scatter.remove()
        
        # Plot new points
        self.scatter_points[plot_index] = []
        self.mean_points[plot_index] = []
        generation = len(self.scatter_points[plot_index])
        
        ymin = np.inf
        ymax = -np.inf
        # Plot individual points
        for i, inner_list in enumerate(generation_data):  # i is the outer list index
            for j, value in enumerate(inner_list):  # j is the index within inner list, value is the y
                scatter = ax.scatter(i, value, cmap='blue') #i is the x, value is the y. only label the first point for each series
                self.scatter_points[plot_index].append(scatter)
                if value < ymin: 
                    ymin = value
                elif value > ymax:
                    ymax = value
            mean_scatter = ax.scatter(i, np.mean(inner_list), cmap='red')
            self.mean_points[plot_index].append(mean_scatter)
        
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Example usage
if __name__ == "__main__":
    # Create display with 1 scatter plot
    display = DisplayManager(num_images=1, num_plots=1)
    
    # Simulate updating with new generations
    for generation in range(5):
        # Simulate some fitness values
        fitness_values = np.random.normal(50 + generation * 5, 10, 20)
        display.update_scatter(0, fitness_values)

        img1 = np.random.rand(100, 100)
        display.update_image(0, img1)

        plt.pause(1)  # Pause to see the update
    
    plt.show()
