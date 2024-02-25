import numpy as np
import matplotlib.pyplot as plt
from Read_data import CSVReader
import inspect

# Usage
directory = "C:\\Users\\baspi\\OneDrive\\Masaüstü\\MaterialData\\DataFile"
csv_reader = CSVReader(directory)
df = csv_reader.read_and_save_csv()


class ReflectanceCalculator:
    """
    A class to calculate and visualize reflectance.

    Attributes:
        df (DataFrame): The DataFrame containing wavelength and refractive index data.
        materials (list): A list of material names.
    """

    def __init__(self, directory):
        """
        Initializes the ReflectanceCalculator with directory to read CSV data.

        Args:
            directory (str): The directory containing CSV files.
        """
        self.csv_reader = CSVReader(directory)
        self.df = self.csv_reader.read_and_save_csv()

    def interpolate_wavelength(self, wavelengths, refractive_indices, bottom_limit, upper_limit, num_points):
        """
        Interpolates wavelengths and refractive indices.

        Args:
            wavelengths (numpy.ndarray): Array of wavelengths.
            refractive_indices (numpy.ndarray): Array of refractive indices.
            bottom_limit (float): Lower limit of interpolation range.
            upper_limit (float): Upper limit of interpolation range.
            num_points (int): Number of points to interpolate.

        Returns:
            tuple: Interpolated wavelengths and refractive indices.
        """
        # Generate an array of desired wavelengths
        desired_wavelengths = np.linspace(bottom_limit, upper_limit, num_points)

        # Perform linear interpolation within the range of the original data
        interpolated_refractive_indices = np.interp(desired_wavelengths, wavelengths, refractive_indices)

        # Use the first 10 data points for linear extrapolation between bottom_limit and min(wavelengths)
        first_10_wavelengths = wavelengths[:10]
        first_10_refractive_indices = refractive_indices[:10]

        p_first = np.polyfit(first_10_wavelengths, first_10_refractive_indices, 1)  # Fit a linear polynomial
        extrapolated_first = np.polyval(p_first, desired_wavelengths[desired_wavelengths < wavelengths[0]])

        # Use the last 10 data points for linear extrapolation between max(wavelengths) and upper_limit
        last_10_wavelengths = wavelengths[-10:]
        last_10_refractive_indices = refractive_indices[-10:]

        p_last = np.polyfit(last_10_wavelengths, last_10_refractive_indices, 1)  # Fit a linear polynomial
        extrapolated_last = np.polyval(p_last, desired_wavelengths[desired_wavelengths > wavelengths[-1]])

        # Ensure sizes match and concatenate extrapolated and interpolated data
        final_refractive_indices = np.concatenate((extrapolated_first,
                                                   interpolated_refractive_indices,
                                                   extrapolated_last))

        return desired_wavelengths, final_refractive_indices

    def get_info(self, material_names, bottom=0.3, upper=0.9, datapoints=100):
        """
        Retrieves interpolated data for specified materials.

        Args:
            material_names (list): List of material names.
            bottom (float): Bottom limit for interpolation.
            upper (float): Upper limit for interpolation.
            datapoints (int): Number of data points.

        Returns:
            dict: A dictionary containing interpolated data for each material.
        """
        data_dict = {}
        for material in material_names:
            # Extract wavelength and refractive index data for the current material
            wavelengths = np.array(df[material].iloc[:, 0], dtype=float)
            refractive_indices = np.array(df[material].iloc[:, 1], dtype=float)

            # Interpolate wavelengths and refractive indices
            bottom_limit = bottom
            upper_limit = upper
            num_points = datapoints
            interpolated_wavelengths, interpolated_refractive_indices = interpolate_wavelength(
                wavelengths, refractive_indices, bottom_limit, upper_limit, num_points)

            data_dict[material] = (interpolated_wavelengths, interpolated_refractive_indices)

        return data_dict

    def create_matrix(self, materials, thickness, upper, bottom, data_points):
        """
        Creates a matrix for reflectance calculation.

        Args:
            materials (list): List of materials.
            thickness (list): List of thickness values.
            upper (float): Upper limit for wavelength.
            bottom (float): Bottom limit for wavelength.
            data_points (int): Number of data points.

        Returns:
            list: List of reflectance values.
        """

        lamda = np.linspace(bottom, upper, data_points)  # Create wavelength data with the increment of 0.01 nm
        gammai = get_info(df, materials, upper=upper, bottom=bottom,
                          datapoints=data_points)  # it will take refractive index value of each material
        gamma_0 = 1.0  # basicly n0 = 1
        gamma_s = 1.52  # ns = 1.52     # Later, these values will be taken from user.
        thickness = thickness  # Thickness information of each layer

        R = []  # List for reflectance values (between 0 and 1)
        for wavelentgh in range(len(lamda)):
            MT = np.identity(2)  # Initiate with identity matrix.
            for material in range(len(materials)):
                phase = (2 * np.pi / float(lamda[wavelentgh])) * float(
                    gammai[materials[material]][1][wavelentgh]) * float(thickness[material])

                M1 = np.array(
                    [[np.cos(phase), complex(0, np.sin(phase)) / float(gammai[materials[material]][1][wavelentgh])],
                     [float(gammai[materials[material]][1][wavelentgh]) * complex(0, np.sin(phase)), np.cos(phase)]],
                    dtype=complex)

                MT = np.dot(MT, M1)  # Calculate MT = M1*M2*M3 ... Mn

            ri = (((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) - (MT[1, 0]) - (
                        float(gamma_s) * MT[1, 1])) /
                  ((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) + (MT[1, 0]) + (
                              float(gamma_s) * MT[1, 1])))
            # Calculate r using the formula 19-36.
            R.append((np.dot(ri, ri.conjugate()).real))

        return R

    def plotGraph(self):
        """
        Plots the reflectance graph.

        Returns:
            numpy.ndarray: Array of wavelength values.
        """
        # Generate an array of wavelengths using NumPy
        lambda_values = np.arange(float(df["Wavelength >"][0]), float(df["Wavelength <"][0]), 0.01)
        # In my design, I will just consider the given wavelengths in first row. It will change later when i change the design.

        # Reflectance values taken from create_matrix function.
        reflectance_values = np.multiply(create_matrix(df), 100)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the data on the axis
        ax.plot(lambda_values, reflectance_values, "-r")

        # Set titles and labels
        ax.set_title("Wavelength vs Reflectance")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance (Percentage)")

        # Add a grid
        ax.grid()

        # Show the plot
        plt.show()
        return lambda_values

    def help(self, method_name=None):
        """
        Display help information about the methods of ReflectanceCalculator class.

        Args:
            method_name (str, optional): The name of the method to get help for.
                If None, help for all methods will be provided.

        Returns:
            str: Help information about the specified method or all methods.
        """
        if method_name:
            # Get the method object from the class
            method = getattr(self, method_name, None)
            if method is None or not inspect.ismethod(method):
                return f"Method '{method_name}' not found or is not a method."
            else:
                # Return the docstring of the method
                return inspect.getdoc(method)
        else:
            # Get all methods of the class
            methods = [name for name, _ in inspect.getmembers(self, predicate=inspect.ismethod)]
            # Construct a string with docstrings of all methods
            methods_docstrings = "\n\n".join(
                f"{method}:\n{inspect.getdoc(getattr(self, method))}" for method in methods)
            return f"All methods:\n{methods_docstrings}"

rc = ReflectanceCalculator(directory)
print(rc.help())  # Display help information

