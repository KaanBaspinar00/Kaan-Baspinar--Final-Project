# Thin Film Layer Reflectance Calculation

This Python module provides functions to calculate and visualize the relationship between wavelength and reflectance for a stack of thin film layers.

## Table of Contents
- [Functions](#functions)
  - [plotGraph](#plotgraph)
  - [create_matrix](#create_matrix)
- [Usage](#usage)
- [Examples](#examples)

## Functions

### plotGraph

The `plotGraph` function is designed to plot the graph showing the relationship between wavelength and reflectance. It takes a DataFrame as input, which should contain information about different layers such as material, thickness, wavelength range, and angle.

### create_matrix

The `create_matrix` function is used to create a transfer matrix (MT) and calculate the reflectance (R) value for each wavelength in the provided range. It takes a DataFrame as input, containing information about the material, thickness, wavelength range, and angle for each layer.

## Usage

1. **Input Material Data:**
   - Create a DataFrame with information about the thin film layers, including material, thickness, wavelength range, and angle.

2. **Call `create_matrix` Function:**
   - Use the `create_matrix` function with the DataFrame as input to generate a list of reflectance values.

3. **Call `plotGraph` Function:**
   - Use the `plotGraph` function with the DataFrame as input to visualize the relationship between wavelength and reflectance.

```python
import pandas as pd
from FunctionsForInterfaceQt5.py import create_matrix, plotGraph

# Example DataFrame
df = pd.DataFrame({
    "Material": ["material1", "material2"],
    "Thickness": [100, 200],
    "Wavelength >": [300, 400],
    "Wavelength <": [800, 900],
    "Angle": [0, 0]
})

# Calculate reflectance values
reflectance_values = create_matrix(df)

# Plot the graph
plotGraph(df)
