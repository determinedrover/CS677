# Volume Rendering Using Ray Casting

This project implements volume rendering using ray casting for converting 3D datasets into 2D images. The rendering process has been tested on multiple cores with varying step sizes. For detailed information, refer to the [project report](xyz.abc).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Volume rendering using ray casting is a technique to visualize 3D data by projecting it into 2D images. This project processes volumetric data through ray casting to produce 2D visualizations, making it easier to analyze and interpret complex 3D structures.

## Features

- **Ray Casting:** Efficient ray casting implementation for rendering 3D datasets into 2D images.
- **Transfer Functions:** Includes opacity and color transfer functions to map data values to colors and opacities.
- **Parallel Processing:** Utilizes MPI for parallel processing across multiple cores to speed up rendering.
- **Data Handling:** Supports reading and processing volumetric data from raw files with specific dimension extraction.

## Prerequisites

Before running the code, ensure the following software is installed:

1. OpenCV (version 4.0 or higher)
2. MPI (Message Passing Interface)


## Installation

1. If OpenCV is not already installed, you can install it using the following commands:
    ```bash
    sudo apt-get update
    sudo apt-get install libopencv-dev
    ```
2. Install MPI:
    ```bash
    sudo apt-get install mpich
    ```

## Usage

1. To compile the code, use the following command. Make sure to adjust the OpenCV include and library paths according to your installation if not in the standard directories:
    ```bash
    mpicxx -std=c++11 -o executable main.cpp \
    -I/path/to/opencv/include \
    -L/path/to/opencv/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lmpi
    ```
   
2. To run the executable, use the following command:
    ```bash
    mpirun -np 4 ./executable Isabel_1000x1000x200_float32.raw 1 0.75 0 999 0 1000
    ```


## Dependencies

- C++ Compiler (supporting C++11 or later)
- [MPI](https://www.open-mpi.org/) (Open MPI or MPICH)
- [OpenCV](https://opencv.org/) (for saving rendered images)
- [Isabel_High_Resolution_Raw](https://iitk-my.sharepoint.com/:f:/g/personal/soumyad_iitk_ac_in/Eij8743ERAhNpFD2ZtbMjMQBguEnvAK4Nd6OTdWNOhYv4A?e=Q0yqYY) (for dataset provided in the assignment)
  
## Group Info

This project was part of the Assignment-1 for the Course CS677:TOPICS IN LARGE DATA ANALYSIS AND VISUALIZATION 

Group 10:
1. Lakshvant Balachandran - 210557
2. Parthapratim Chatterjee - 210705
3. Divyansh Mittal - 210358
