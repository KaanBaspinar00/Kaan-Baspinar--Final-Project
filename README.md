# Kaan-Baspinar--Final-Project

# Interface for Data Input and Visualization

This Python program provides a graphical user interface (GUI) for users to input data related to material layers and visualize the information using Matplotlib.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

This program is designed to facilitate the input of material layer data through a user-friendly interface. Users can input information such as material, thickness, wavelength range, and angle for each layer. The interface allows for the dynamic addition and deletion of layers, making it easy to manage complex data sets.

## Features

- Intuitive graphical user interface.
- Dynamic addition and deletion of layers.
- Visualization of input data using Matplotlib.

## Usage

1. **Input Material Data:**
   - Fill in the material, thickness, wavelength range, and angle for each layer in the table.
   - Use the "Add Line" button (or press `Ctrl + A`) to add a new row for each layer.

2. **Visualize Data:**
   - After inputting data for all layers, press the "Show Graph" button (or press `Ctrl + G`) to visualize the data.
   - The program will generate a graph based on the provided information.

3. **Delete Rows:**
   - To delete a layer, select the corresponding row and press `Ctrl + Delete`.

4. **Unit of Length:**
   - Ensure that all length values are provided in nanometers.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KaanBaspinar00/Kaan-Baspinar--Final-Project.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Kaan-Baspinar--Final-Project
    ```

3. Run the program:

    ```bash
    python Tasarla1(Nice).py
    ```

## Dependencies

- PyQt5
- Matplotlib
- Numpy
- Pandas

Install dependencies using:

```bash
pip install PyQt5 matplotlib numpy pandas

