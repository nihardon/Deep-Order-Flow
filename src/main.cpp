#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <unistd.h> // For reading from Stdin


struct Matrix {
    std::vector<float> data;
    int rows;
    int cols;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}
    float& at(int r, int c) { return data[r * cols + c]; }
    
    static Matrix multiply(Matrix& A, Matrix& B) {
        Matrix C(A.rows, B.cols);
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < A.cols; k++) sum += A.at(i, k) * B.at(k, j);
                C.at(i, j) = sum;
            }
        }
        return C;
    }
    void relu() { for (float& val : data) if (val < 0) val = 0; }
    void softmax() {
        float max_val = *std::max_element(data.begin(), data.end());
        float sum = 0.0f;
        for (float& val : data) { val = std::exp(val - max_val); sum += val; }
        for (float& val : data) val /= sum;
    }
};

std::vector<float> load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + filename);
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) throw std::runtime_error("Read error");
    return buffer;
}

int main() {
    // Turn off synchronization with C-style I/O for speed
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    try {
        // Load Weights
        std::vector<float> raw_weights = load_weights("../data/model_weights.bin");
        int input_dim = 140; 
        int hidden_size = 16;
        
        // Prepare Matrices
        Matrix W1(input_dim, hidden_size);
        for (int i=0; i < input_dim*hidden_size; i++) W1.data[i] = raw_weights[i];
        
        Matrix W2(hidden_size, 3);
        // Offset for second layer weights
        int offset = input_dim*hidden_size; 

        for (int i=0; i < hidden_size*3; i++) W2.data[i] = raw_weights[offset + i];

        std::cerr << "--- HFT ENGINE ONLINE & WAITING FOR DATA ---" << std::endl;

        // We read raw binary bytes from stdin
        std::vector<float> input_buffer(input_dim);
        
        while (true) {

            std::cin.read(reinterpret_cast<char*>(input_buffer.data()), input_dim * sizeof(float));
            
            if (std::cin.gcount() != input_dim * sizeof(float)) {
                break; 
            }

            // Inference
            Matrix input(1, input_dim);
            input.data = input_buffer; 

            Matrix hidden = Matrix::multiply(input, W1);
            hidden.relu();
            
            Matrix output = Matrix::multiply(hidden, W2);
            output.softmax();

            // Output
            float p_down = output.data[0];
            float p_hold = output.data[1];
            float p_up   = output.data[2];
            
            std::string signal = "HOLD";
            if (p_up > 0.6) signal = "BUY  (UP)";
            if (p_down > 0.6) signal = "SELL (DOWN)";

            std::cerr << "\r[LIVE] BTC/USD | Down: " << std::fixed << std::setprecision(2) << p_down 
                      << " | Hold: " << p_hold 
                      << " | Up: " << p_up 
                      << " | Signal: " << signal << "      " << std::flush;
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}