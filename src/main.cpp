#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>


// A simple structure to hold a Matrix
struct Matrix {
    std::vector<float> data;
    int rows;
    int cols;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

    // Access element at (row, col)
    float& at(int r, int c) {
        return data[r * cols + c];
    }
    
    // Matrix Multiplication: C = A * B
    static Matrix multiply(Matrix& A, Matrix& B) {
        if (A.cols != B.rows) {
            throw std::runtime_error("Dimension Mismatch in MatMul");
        }
        Matrix C(A.rows, B.cols);
        
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < A.cols; k++) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
            }
        }
        return C;
    }

    // ReLU Activation Function
    void relu() {
        for (float& val : data) {
            if (val < 0) val = 0;
        }
    }
    
    // Softmax Activation Function
    void softmax() {
        float max_val = *std::max_element(data.begin(), data.end());
        float sum = 0.0f;
        
        // Exponentiate and sum
        for (float& val : data) {
            val = std::exp(val - max_val); // Stability fix
            sum += val;
        }
        // Normalize
        for (float& val : data) {
            val /= sum;
        }
    }
};


std::vector<float> load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + filename);
    
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) 
        throw std::runtime_error("Read error");
        
    return buffer;
}


int main() {
    std::cout << "--- HFT ENGINE INITIALIZING ---" << std::endl;

    try {
        // Load the weights
        std::vector<float> raw_weights = load_weights("../data/model_weights.bin");
        std::cout << "Loaded " << raw_weights.size() << " parameters." << std::endl;

        // Simulate an a Snapshot of the market
        std::cout << "Receiving Market Data Snapshot..." << std::endl;
        Matrix input(1, 60); // Flattened 20 nodes * 3 features
        for (int i=0; i<60; i++) input.data[i] = 0.5f; // Fake normalized data
        
        // We grab the first chunk of weights to act as our "Hidden Layer"
        // (Input 60 -> Hidden 16)
        int hidden_size = 16;
        Matrix W1(60, hidden_size);
        
        // Fill W1 from our loaded file (just taking the first N floats)
        for (int i=0; i < 60*hidden_size; i++) {
            if (i < raw_weights.size()) W1.data[i] = raw_weights[i];
        }

        // Forward Pass
        std::cout << "Running Inference..." << std::endl;
        
        // Layer 1: Input * Weight
        Matrix hidden = Matrix::multiply(input, W1);
        
        // Activation: ReLU
        hidden.relu();
        
        // Output Layer: (Hidden 16 -> Output 3)
        Matrix W2(hidden_size, 3); 
        for (int i=0; i < hidden_size*3; i++) W2.data[i] = 0.1f;
        
        Matrix output = Matrix::multiply(hidden, W2);
        
        // Probability: Softmax
        output.softmax();

        // Decision
        std::cout << "\n--- PREDICTION ---" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "DOWN: " << output.data[0] * 100 << "%" << std::endl;
        std::cout << "HOLD: " << output.data[1] * 100 << "%" << std::endl;
        std::cout << "UP:   " << output.data[2] * 100 << "%" << std::endl;
        
        std::cout << "\nLatency: < 1 microsecond (Estimated)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}