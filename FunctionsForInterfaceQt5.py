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

#df = pd.DataFrame({"Material": [2.1],"Thickness": [100], "Wavelength >": [300], "Wavelength <": [800], "Angle": [0]})

#create_matrix(df)
#plotGraph(df)
"""
for i in range(50,600):
# Given values
    x = np.arange([50,600])
    n1 = 2.1
    gamma_0 = 1  # Your value for gamma_0
    gamma_s = 1.5
    ns = 1.5  # Your value for gamma_s
    phase = (2 * np.pi / i) * n1 * 40

    m11 = np.cos(phase)
    m12 = complex(0, np.sin(phase)) / n1
    m21 = complex(0, np.sin(phase)) * n1
    m22 = np.cos(phase)

    Mi = np.array([[m11, m12], [m21, m22]])
    Identity = Mi
    r = np.divide(
        ((gamma_0 * Identity[0, 0]) + (gamma_0 * gamma_s * Identity[0, 1]) - (Identity[1, 0]) - (gamma_s * Identity[1, 1])),
        ((gamma_0 * Identity[0, 0]) + (gamma_0 * gamma_s * Identity[0, 1]) + (Identity[1, 0]) + (gamma_s * Identity[1, 1])))
    print("\noooo\nr: ",r*r.conjugate())
"""

"""
def CreateMT(df):
    lamda = np.arange(float(df["Wavelength >"][2]), float(df["Wavelength <"][2]), 0.05)
    gammai = df["Material"]
    gamma_0 = 1
    gamma_s = 1.5
    thickness = df["Thickness"]
    R = []
    for wavelentgh in lamda:
        MT = np.identity(2)
        for i in range(0,len(gammai)):
            phase = (2 * np.pi / float(wavelentgh)) * float(gammai[i]) * float(thickness[i])
            M1 = np.array([[np.cos(phase), complex(np.sin(phase)) / float(gammai[i])],
                           [float(gammai[i]) * complex(np.sin(phase)), np.cos(phase)]])
            MT = np.dot(M1,MT)

        ri = np.divide(((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) - (MT[1, 0]) -
                       (float(gamma_s) * MT[1, 1])), ((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) +
                       (MT[1, 0]) + (gamma_s * MT[1, 1])))
        R.append((ri**2).real)

    return R"""



