import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class DisplayManager:
    def __init__(self, num_images=0, num_plots=1, figsize=None):
        # Calculate default figure size if not provided
        if figsize is None:
            figsize = (5 * (num_images + num_plots), 5)
        
        # Create figure and subplots
        self.fig, axes = plt.subplots(1, num_images + num_plots, figsize=figsize)
        # Ensure axes is always a list even with single subplot
        self.axes = [axes] if num_images + num_plots == 1 else axes
        
        self.scatter_points = {}  # Store scatter plot points
        self.mean_points = {}    # Store mean points
        
        self.num_images = num_images

        # Initialize displays
        for i, ax in enumerate(self.axes):
            if i < num_images:
                # Initialize image display
                display = ax.imshow(np.random.uniform(size=(100, 100)).astype(float),
                                    cmap="twilight_shifted")
                plt.colorbar(display, ax=ax)
                #ax.set_title(f'Image {i+1}')
            else:
                # Initialize scatter plot
                ax.set_title('Fitness History')
                ax.set_xlabel('Generation')
                ax.ylabel = 'Fitness'
                ax.grid(True)
                
                # Store the axis for later use
                plot_index = i - num_images
                self.scatter_points[plot_index] = []  # List to store scatter points
                self.mean_points[plot_index] = []     # List to store mean points
        
        plt.tight_layout()
        plt.show(block=False)

    def update_image(self, index, new_image):
        """Update image at given index"""
        self.axes[index].images[0].set_array(new_image)
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
        
        # Update axis limits if needed
        #ax.set_xlim(-0.5, generation + 0.5)
        #ax.set_xticks(range(generation + 1))
        
        # Update y limits if needed
        # all_values = [val for gen in generation_data for val in (gen if isinstance(gen, list) else [gen])]
        # if all_values:
        #     ymin, ymax = min(all_values), max(all_values)
        margin = (ymax - ymin) * 0.1  # Add 10% margin
        ax.set_ylim(ymin - margin, ymax + margin)
        
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
