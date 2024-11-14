
#include <codecvt>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <regex>
#include<cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// Define constants
#define TERMINATION_THRESHOLD 1.0f
using namespace std;
// Function to read the dataset and reshape it
void readDataset(const std::string& filename, std::vector<float>& data, int& xDim, int& yDim, int& zDim) {
    // Extract dimensions from the filename using regex
    std::regex pattern(R"(Isabel_(\d+)x(\d+)x(\d+)_float32\.raw)");
    std::smatch match;
    
    if (!std::regex_match(filename, match, pattern)) {
        throw std::runtime_error("Filename does not match expected format.");
    }

    try {
        xDim = std::stoi(match[1].str());
        yDim = std::stoi(match[2].str());
        zDim = std::stoi(match[3].str());
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Error parsing dimensions from filename.");
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open dataset file.");
    }

    size_t totalSize = static_cast<size_t>(xDim) * yDim * zDim;
    data.resize(totalSize);

    file.read(reinterpret_cast<char*>(data.data()), totalSize * sizeof(float));
    if (!file) {
        throw std::runtime_error("Error reading data from file.");
    }

    file.close();
}


float opacityTF(float a){
 float intensity[] = {-4926.59326171875, -1310.073486328125, 103.00382995605469, 2593.85205078125};
    float opacity[] = {1.0, 0.75641030073165894, 0.0, 0.053846154361963272};

    int n = sizeof(intensity) / sizeof(intensity[0]);

    // Handle edge cases where 'a' is outside the given range
    if (a <= intensity[0]) {
       
        return opacity[0];  // Return max opacity for the lowest intensity
    }
    if (a >= intensity[n-1]) {

        return opacity[n-1];  // Return min opacity for the highest intensity
    }

    // Linear interpolation
    for (int i = 0; i < n - 1; ++i) {
        if (a >= intensity[i] && a <= intensity[i+1]) {
            // Perform linear interpolation
            float t = (a - intensity[i]) / (intensity[i+1] - intensity[i]);
            return opacity[i] + t * (opacity[i+1] - opacity[i]);
        }
    }

    // Should never reach here
    return 0.0f;
}

std::vector<float> colorTF(float a) {
    // Data points: intensity and corresponding (R, G, B) colors (normalized)
     float intensity[] = {
        -4926.59, -4803.33, -4680.07, -4556.81, -4433.55, -4310.28, -4187.02, -4063.76,
        -3940.5, -3817.24, -3693.97, -3570.71, -3447.45, -3324.19, -3200.93, -3077.66,
        -2954.4, -2831.14, -2707.88, -2584.62, -2466.28, -2461.35, -2461.28, -2461.28,
        -2333.28, -2205.27, -2077.27, -1949.26, -1821.26, -1693.26, -1565.25, -1437.25,
        -1309.24, -1181.24, -1053.24, -925.233, -797.229, -669.226, -541.222, -413.218,
        -285.214, -157.21, -29.2065, 98.8726, 348.371, 597.868, 847.366, 1096.86, 1346.36,
        1595.86, 1845.36, 2094.86, 2344.35, 2469.1, 2593.85
    };

     float R[] = {
        0.301961, 0.396078, 0.494118, 0.588235, 0.662745, 0.741176, 0.788235, 0.862745,
        0.901961, 0.917647, 0.925490, 0.937255, 0.945098, 0.952941, 0.964706, 0.968627,
        0.972549, 0.980392, 0.980392, 0.988235, 0.984314, 0.988235, 0.952941, 0.952941,
        0.890196, 0.827451, 0.776471, 0.725490, 0.678431, 0.631373, 0.580392, 0.537255,
        0.498039, 0.462745, 0.431373, 0.403922, 0.372549, 0.345098, 0.317647, 0.286275,
        0.254902, 0.231373, 0.200000, 0.149020, 0.200000, 0.247059, 0.305882, 0.372549,
        0.443137, 0.517647, 0.600000, 0.686275, 0.760784, 0.807843, 0.890196
    };

    float G[] = {
        0.047059, 0.0392157, 0.054902, 0.113725, 0.168627, 0.227451, 0.290196, 0.380392,
        0.458824, 0.521569, 0.580392, 0.643137, 0.709804, 0.768627, 0.827451, 0.878431,
        0.917647, 0.949020, 0.972549, 0.988235, 0.988235, 0.988235, 0.952941, 0.952941,
        0.890196, 0.823529, 0.764706, 0.713726, 0.662745, 0.607843, 0.556863, 0.505882,
        0.458824, 0.419608, 0.388235, 0.356863, 0.321569, 0.294118, 0.262745, 0.231373,
        0.200000, 0.172549, 0.145098, 0.196078, 0.254902, 0.317647, 0.388235, 0.458824,
        0.533333, 0.615686, 0.698039, 0.784314, 0.858824, 0.901961, 0.956863
    };

    float B[] = {
        0.090196, 0.0588235, 0.0352941, 0.0235294, 0.0156863, 0.00392157, 0.000000,
        0.0117647, 0.027451, 0.0470588, 0.0784314, 0.121569, 0.184314, 0.247059,
        0.325490, 0.423529, 0.513726, 0.596078, 0.670588, 0.756863, 0.854902,
        0.858824, 0.894118, 0.894118, 0.807843, 0.737255, 0.678431, 0.627451,
        0.580392, 0.533333, 0.486275, 0.443137, 0.407843, 0.372549, 0.345098,
        0.317647, 0.294118, 0.266667, 0.239216, 0.211765, 0.184314, 0.164706,
        0.137255, 0.278431, 0.345098, 0.415686, 0.494118, 0.568627, 0.643137,
        0.725490, 0.800000, 0.870588, 0.929412, 0.960784, 0.984314
    };

    int n = sizeof(intensity) / sizeof(intensity[0]);

    // If a is out of bounds, return the first or last color
    if (a <= intensity[0]) return std::vector<float>({(R[0]), (G[0]), (B[0])});
    if (a >= intensity[n - 1]) return std::vector<float>({(R[n - 1]), (G[n - 1]), (B[n - 1])});

    // Find the appropriate interval for interpolation
    for (int i = 0; i < n - 1; ++i) {
        if (a >= intensity[i] && a <= intensity[i + 1]) {
            // Linear interpolation factor
            float t = (a - intensity[i]) / (intensity[i + 1] - intensity[i]);
            
            // Interpolate each RGB component
            float red = ((R[i] + t * (R[i + 1] - R[i])));
            float green = ((G[i] + t * (G[i + 1] - G[i])));
            float blue = ((B[i] + t * (B[i + 1] - B[i])));
            
            return std::vector<float>{red, green, blue};
        }
    }

    // Default return, though this shouldn't happen
    return std::vector<float>{0, 0, 0};
}
void saveImageFromVector(const std::vector<float>& data, int width, int height, int channels, const std::string& filename) {
    // Ensure the data is normalized to 0-255 range
    std::vector<u_char> imageData(data.size());
    float minVal = *std::min_element(data.begin(), data.end());
    float maxVal = *std::max_element(data.begin(), data.end());

    for (size_t i = 0; i < data.size(); ++i) {
        imageData[i] = static_cast<u_char>(255 * (data[i] - minVal) / (maxVal - minVal));
    }

    // Create OpenCV Mat object
    cv::Mat image(height, width, CV_8UC(channels), imageData.data());
    // cv::Mat rgbImage;
    // cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);   

    // Save the image
    if (!cv::imwrite(filename,image)) {
        std::cerr << "Error: Could not write image to file " << filename << std::endl;
    }
}





void rayCasting(int xStart, int xEnd, int yStart, int yEnd,int zDim,int yDim,int xDim, 
                const std::vector<float>& data, float stepSize, 
                std::vector<float>& result, int& earlyTerminatedRays) {
    int xSize = xEnd - xStart + 1;
    int ySize = yEnd - yStart + 1;
    int totalRays = xSize * ySize;
    earlyTerminatedRays = 0;

    // Resize result vector to store RGB values for each ray
    result.resize(totalRays * 3, 0.0f);

    for (int x = xStart; x <= xEnd; ++x) {
        for (int y = yStart; y <= yEnd; ++y) {
            float accumulatedValue = 0.0f;
            float r = 0, g = 0, b = 0;

            // Start ray marching along the Z-axis
            for (float z = 0; z < zDim; z+=stepSize) {
                // Compute the index in the flattened array (considering the full dimensions)
                int idx = x + y * xDim + z * yDim * xDim;
                int zIdx = static_cast<int>(z);  // Integer part (grid point)
                int zNext = zIdx + 1;  // Next grid point

                if (zNext >= zDim) break; // Avoid accessing out of bounds

                // Calculate the interpolation factor 't'
                float t = (z - zIdx) / (zNext - zIdx);

                // Interpolate the data between the two z indices
                int idx1 = x + y * xDim + zIdx * yDim * xDim;
                int idx2 = x + y * xDim + zNext * yDim * xDim;

                float interpolatedData = data[idx1] + t * (data[idx2] - data[idx1]);

                
                // Get RGB and opacity values from the transfer functions
                std::vector<float> rgb = colorTF(interpolatedData);
                float op = opacityTF(interpolatedData);

                // For the first point along the Z direction
                if (z == 0) {
                    r = rgb[0];
                    g = rgb[1];
                    b = rgb[2];
                    accumulatedValue = op;
                } else {
                    // Accumulate the RGB values with front-to-back compositing
                    r = r + ((1 - accumulatedValue) * rgb[0] * op);
                    g = g + ((1 - accumulatedValue) * rgb[1] * op);
                    b = b + ((1 - accumulatedValue) * rgb[2] * op);
                    
                    // Accumulate the opacity value
                    accumulatedValue = accumulatedValue + (1 - accumulatedValue) * op;
                }

                // Check for early termination
                if (accumulatedValue >= TERMINATION_THRESHOLD) {
                    earlyTerminatedRays++;
                    break;
                }
            }

            // Store the final RGB values for this ray in the result array
            int index = (x - xStart) + (y - yStart)*xSize;
            result[index * 3] = static_cast<float>(b);
            result[index * 3 + 1] = static_cast<float>(g);
            result[index * 3 + 2] = static_cast<float>(r);
        }
    }
}


void copySubResultToFinal(int xStart, int yStart, const std::vector<float>& subResult, 
                          std::vector<float>& finalResult, int xBoundMin,int xBoundMax, int yBoundMin,int yBoundMax, 
                          int xSize, int ySize, int zDim) {

    for (int i = 0; i < xSize; ++i) {
        for (int j = 0; j < ySize; ++j) {
            int finalIndex = ((xStart - xBoundMin + i) + (yStart - yBoundMin + j)*(xBoundMax-xBoundMin+1)) * 3;
            int subIndex = ( i+ j*xSize) * 3;
            finalResult[finalIndex] = subResult[subIndex];
            finalResult[finalIndex + 1] = subResult[subIndex + 1];
            finalResult[finalIndex + 2] = subResult[subIndex + 2];
        }
    }
}


// Function to transpose the image data
std::vector<float> transposeImage(const std::vector<float>& input, int width, int height, int channels) {
    std::vector<float> transposed(width * height * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                // Calculate the index in the original image
                int originalIndex = (y * width + x) * channels + c;
                // Calculate the index in the transposed image
                int transposedIndex = (x * height + y) * channels + c;
                transposed[transposedIndex] = input[originalIndex];
            }
        }
    }
    return transposed;
}
std::vector<float> flipAndTransposeImage(const std::vector<float>& input, int width, int height, int channels) {
    std::vector<float> flippedAndTransposed(width * height * channels);

    // Compute the maximum x and y bounds
    int maxX = width - 1;
    int maxY = height - 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                // Calculate the index in the original image
                int originalIndex = (y * width + x) * channels + c;
                // Calculate the new flipped coordinates
                int flippedX = maxX - x;
                int flippedY = maxY - y;
                // Calculate the index in the flipped image
                int flippedIndex = (flippedY * width + flippedX) * channels + c;
                flippedAndTransposed[flippedIndex] = input[originalIndex];
            }
        }
    }
    return flippedAndTransposed;
}

// Function to perform domain decomposition and ray casting
void volumeRendering(MPI_Comm comm, int xDim, int yDim, int zDim, const std::vector<float>& data, 
                     float stepSize, int xBoundMin, int xBoundMax, int yBoundMin, int yBoundMax, 
                     int decompositionType) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int xStart, xEnd, yStart, yEnd;
    
    // 1D partitioning
    if (decompositionType == 1) {
        int xChunkSize = (xBoundMax - xBoundMin + 1) / size;
        xStart = xBoundMin + rank * xChunkSize;
        xEnd = (rank == size - 1) ? xBoundMax : xStart + xChunkSize - 1;
        yStart = yBoundMin;
        yEnd = yBoundMax;
    }
    // 2D partitioning
    else if (decompositionType == 2) {
        int procX = std::ceil(std::sqrt(size));
        int procY = size / procX;
        
        int xChunkSize = (xBoundMax - xBoundMin + 1) / procX;
        int yChunkSize = (yBoundMax - yBoundMin + 1) / procY;
        
        int xRank = rank % procX;
        int yRank = rank / procX;
        
        xStart = xBoundMin + xRank * xChunkSize;
        xEnd = (xRank == procX - 1) ? xBoundMax : xStart + xChunkSize - 1;
        
        yStart = yBoundMin + yRank * yChunkSize;
        yEnd = (yRank == procY - 1) ? yBoundMax : yStart + yChunkSize - 1;
    }

    // Allocate memory for the result of this subdomain
    int xSize = xEnd - xStart + 1;
    int ySize = yEnd - yStart + 1;
    std::vector<float> subResult(xSize * ySize * 3, 0.0f);

    // Perform ray casting on the subdomain
    int earlyTerminatedRays = 0;
    // measure time 
    auto start1 =  std::chrono::high_resolution_clock::now();
    rayCasting(xStart, xEnd, yStart, yEnd, zDim,yDim,xDim, data, stepSize, subResult, earlyTerminatedRays);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    //calculate maximum render time
    double localTime1 = elapsed1.count();
    double maxTime1;
    MPI_Reduce(&localTime1, &maxTime1, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if(rank == 0){
        cout << "Maximum time taken to render the subdomain: " << maxTime1 << " seconds" << endl;
    }
    std::cout << "Time taken to render the subdomain: " << rank << " " << elapsed1.count() << " seconds" << std::endl;
    string filename = "output" + to_string(rank) + ".png";
    // std::vector<float> transposedResult2 = transposeImage(subResult, xSize, ySize, 3);
    // std::vector<float> subResult2 = flipAndTransposeImage(transposedResult2, xSize, ySize, 3);
    saveImageFromVector(subResult,xSize,ySize, 3, filename);
    // Prepare for merging the sub-results on rank 0
    if (rank == 0) {
        // Allocate memory for the final result on rank 0
        
        std::vector<float> finalResult((xBoundMax - xBoundMin + 1) * (yBoundMax - yBoundMin + 1) *3, 0.0f);
         
        // Copy the local subResult into the finalResult
        copySubResultToFinal(xStart, yStart, subResult, finalResult, xBoundMin,xBoundMax, yBoundMin,yBoundMax, xSize, ySize,zDim);

        // Receive sub-results from other processes and merge into the final result
        for (int i = 1; i < size; i++) {
            int recvXStart, recvXEnd, recvYStart, recvYEnd;
            
            // Receiving the bounds of the subdomain from process i
            MPI_Recv(&recvXStart, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&recvXEnd, 1, MPI_INT, i, 1, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&recvYStart, 1, MPI_INT, i, 2, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&recvYEnd, 1, MPI_INT, i, 3, comm, MPI_STATUS_IGNORE);

            // Calculate the size of the subdomain
            int recvXSize = recvXEnd - recvXStart + 1;
            int recvYSize = recvYEnd - recvYStart + 1;
            // Allocate space for the received subResult
            std::vector<float> recvSubResult(recvXSize * recvYSize * zDim);

            // Receive the subResult from process i
            MPI_Recv(recvSubResult.data(), recvSubResult.size(), MPI_FLOAT, i, 4, comm, MPI_STATUS_IGNORE);

            // Merge the received subResult into the final result
            copySubResultToFinal(recvXStart, recvYStart, recvSubResult, finalResult, xBoundMin,xBoundMax, yBoundMin,yBoundMax, recvXSize, recvYSize,zDim);
        }
    
        int width = xBoundMax - xBoundMin + 1;
    int height = yBoundMax - yBoundMin + 1;
     
    std::vector<float> transposedResult = transposeImage(finalResult, width, height, 3);
    std::vector<float> finalResult2 = flipAndTransposeImage(transposedResult, width, height, 3);
    // Convert the transposed float vector to unsigned char vector

    // Save the transposed image
    // saveImageFromVector(transposedResult, height, width, 3, "output_transposed.png");
    
    // Save the final image
    saveImageFromVector(finalResult, width, height, 3, "output.png");
    saveImageFromVector(finalResult2, height, width, 3, "output_flipped.png");

    } else {
        // Send subdomain bounds to rank 0
        MPI_Send(&xStart, 1, MPI_INT, 0, 0, comm);
        MPI_Send(&xEnd, 1, MPI_INT, 0, 1, comm);
        MPI_Send(&yStart, 1, MPI_INT, 0, 2, comm);
        MPI_Send(&yEnd, 1, MPI_INT, 0, 3, comm);

        // Send subResult to rank 0
        MPI_Send(subResult.data(), subResult.size(), MPI_FLOAT, 0, 4, comm);
    }

    // Calculate the fraction of rays terminated early
    int totalRays = xSize * ySize;
    float fractionTerminated = static_cast<float>(earlyTerminatedRays) / totalRays;

    // Print the fraction of rays terminated early for each process
    std::cout << "Rank " << rank << ": Fraction of rays terminated early = " << fractionTerminated << std::endl;

    // Timing and output
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Use MPI to get the maximum time across all processes
    double localTime = elapsed.count();
    double maxTime;
    float maxFraction;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&fractionTerminated, &maxFraction, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    if (rank == 0) {

        std::cout << "Total time taken: " << maxTime << " seconds" << std::endl;
        cout << "Fraction of rays terminated early = " << maxFraction/4 << endl;
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 8) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <num_processes> ./executable <dataset> <1D/2D> <step_size> "
                         "<x_min> <x_max> <y_min> <y_max>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // Parse input arguments
    std::string datasetFile = argv[1];
    int decompositionType = std::stoi(argv[2]);
    float stepSize = std::stof(argv[3]);
    int xBoundMin = std::stoi(argv[4]);
    int xBoundMax = std::stoi(argv[5]);
    int yBoundMin = std::stoi(argv[6]);
    int yBoundMax = std::stoi(argv[7]);

    // Variables to store dataset dimensions
    int xDim, yDim, zDim;
    std::vector<float> data;

    if (rank == 0) {
        // Read the dataset on rank 0
        //measure time
        auto start = std::chrono::high_resolution_clock::now();
        readDataset(datasetFile, data, xDim, yDim, zDim);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken to read the dataset: " << elapsed.count() << " seconds" << std::endl;
        
        // Send dataset dimensions to all processes
        MPI_Bcast(&xDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&yDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&zDim, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Send dataset to all processes
        MPI_Bcast(data.data(), xDim * yDim * zDim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        // Receive dataset dimensions from rank 0
        MPI_Bcast(&xDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&yDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&zDim, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocate memory for dataset
        data.resize(xDim * yDim * zDim);
      
        // Receive dataset from rank 0
        MPI_Bcast(data.data(), xDim * yDim * zDim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    }
    //cout << "xDim: " << xDim << " yDim: " << yDim << " zDim: " << zDim << "stepSize" << stepSize << " xBoundMin: " << xBoundMin << " xBoundMax: " << xBoundMax << " yBoundMin: " << yBoundMin << " yBoundMax: " << yBoundMax << " decompositionType: " << decompositionType << endl;
    // Perform volume rendering
       volumeRendering(MPI_COMM_WORLD, xDim, yDim, zDim, data, stepSize, xBoundMin, xBoundMax, yBoundMin, yBoundMax, decompositionType);

    MPI_Finalize();
    return 0;
}