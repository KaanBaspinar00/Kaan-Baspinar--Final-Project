# Some parts of this rewritten in the file Physics_And_Plot.py and some of them can be used later.

# Kaan Başpınar
# e242285@metu.edu.tr

# 20/01/2024

## In this python file, there are two functions.
# plotGraph function is used in the showGraph function in the Tasarla1(Nice).py file.
# It plots the graph to show that the relationship between wavelength and reflectance
# when n different thin film layer is provided.
# createMatrix function is used to create transfer matrix MT and then calculate
# the R value for each wavelength value.
# Then, it returns the list of R values.
###########################ooooo000000000000000000ooooo################################
# 27/01/2024
# 2 functions is added. 

# The function below creates random sized of layers for materials
#                         create_list_of_material(original_array, num_arrays)

# The function below randomly decides thickness  of layers for each materials: 
#                         create_thickness_list(material_list_large,thickness)

###########################ooooo000000000000000000ooooo################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotGraph(df):
    # Generate an array of wavelengths using NumPy
    lambda_values = np.arange(float(df["Wavelength >"][0]), float(df["Wavelength <"][0]), 0.01)
    # In my design, I will just consider the given wavelengths in first row. It will change later when i change the design.

    # Reflectance values taken from create_matrix function.
    reflectance_values = np.multiply(create_matrix(df),100)

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


def create_matrix(df,theta = 0, theta2 = 0):
    lamda = np.arange(float(df["Wavelength >"][0]), float(df["Wavelength <"][0]), 0.01) # Create wavelength data with the increment of 0.01 nm
    gammai = np.asarray(df["Material"]) # it will take refractive index value of each material
    gamma_0 = 1.0 # basicly n0 = 1
    gamma_s = 1.52 # ns = 1.52     # Later, these values will be taken from user.
    thickness = np.asarray(df["Thickness"]) # Thickness information of each layer
    R = [] # List for reflectance values (between 0 and 1)
    for wavelentgh in lamda:
        MT = np.identity(2)   # Initiate with identity matrix.
        for i in range(0,len(gammai)):
            # Calculate phase using the formula
            phase = (2 * np.pi / float(wavelentgh)) * float(gammai[i]) * float(thickness[i])

            M1 = np.array([[np.cos(phase), complex(0,np.sin(phase)) / float(gammai[i])],
                           [float(gammai[i]) * complex(0,np.sin(phase)), np.cos(phase)]],dtype= complex)
            MT = np.dot(MT,M1) # Calculate MT = M1*M2*M3 ... Mn


        ri = (((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) - (MT[1, 0]) - (float(gamma_s) * MT[1, 1]))/
                       ((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) + (MT[1, 0]) + (float(gamma_s) * MT[1, 1])))
        # Calculate r using the formula 19-36.
        R.append((np.dot(ri,ri.conjugate()).real))

    return R
np.random.seed(42)
Materials = np.round(np.random.randint(10,30,20)/10,2)
Thickness = np.arange(200,1020,20)
#print(f"Materials: {Materials}")
#print(f"Thickness: {Thickness}")


def create_list_of_material(original_array, num_arrays):
    new_arrays = []
    np.random.seed(42)
    for _ in range(num_arrays):
        # Randomly choose the size of the new array
        array_size = np.random.randint(1, len(original_array) + 1)

        # Randomly select the first element from the original array
        selected_elements = [np.random.choice(original_array)]

        # Iterate to choose subsequent elements, ensuring they don't repeat successively
        for _ in range(array_size - 1):
            available_elements = np.setdiff1d(original_array, selected_elements[-1])
            selected_elements.append(np.random.choice(available_elements))

        # Create the new array
        new_array = np.array(selected_elements)

        # Append the new array to the list
        new_arrays.append(new_array)

    return new_arrays

material_list = create_list_of_material(Materials,100)

print(f"Material List: \n{material_list}")

def create_thickness_list(material_list_large,thickness):
    large_list_thickness = []
    for little_list in material_list_large:
        little_list_for_thickness = []
        for m in little_list:
            little_list_for_thickness.append(np.random.choice(thickness))
        large_list_thickness.append(np.array(little_list_for_thickness))
    return large_list_thickness

thickness_list = create_thickness_list(material_list,Thickness)
print(f"Thickness List: \n{thickness_list}")

df = pd.DataFrame({"Material": [2.1],"Thickness": [100], "Wavelength >": [300], "Wavelength <": [1000], "Angle": [0]})

create_matrix(df)
#plotGraph(df)
