#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>

// CONFIGURATION
const int INPUT_DIM = 160;   
const int READ_DIM = 161;    
const int HIDDEN_DIM = 16;
const int OUTPUT_DIM = 3;

//  Confidence
const float CONFIDENCE_THRESHOLD = 0.60; 

// Discipline
int COOLDOWN_TICKS = 10;           
const float TAKE_PROFIT = 0.0012;        
const float STOP_LOSS   = 0.0025;        

// AI exit sensitivity
const int CONFIRMATION_STREAK = 8;

struct MarketModel {
    std::vector<float> w1; std::vector<float> b1;
    std::vector<float> w2; std::vector<float> b2;
};

MarketModel load_weights(const std::string& path) {
    MarketModel model;
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Error: No model weights found. Did you run export_weights.py?" << std::endl; exit(1); }
    
    // Resize vectors
    model.w1.resize(HIDDEN_DIM * INPUT_DIM); model.b1.resize(HIDDEN_DIM);
    model.w2.resize(OUTPUT_DIM * HIDDEN_DIM); model.b2.resize(OUTPUT_DIM);
    
    // Read raw bytes
    file.read(reinterpret_cast<char*>(model.w1.data()), model.w1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(model.b1.data()), model.b1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(model.w2.data()), model.w2.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(model.b2.data()), model.b2.size() * sizeof(float));
    return model;
}

// Activation Functions
void relu(std::vector<float>& x) { for (float& val : x) if (val < 0) val = 0; }
void softmax(std::vector<float>& x) {
    float max_val = -1e9, sum = 0;
    for (float val : x) if (val > max_val) max_val = val;
    for (float& val : x) { val = std::exp(val - max_val); sum += val; }
    for (float& val : x) val /= sum;
}

int main() {
    MarketModel model = load_weights("../data/model_weights.bin");
    std::vector<float> input_buffer(READ_DIM);
    std::vector<float> hidden(HIDDEN_DIM);
    std::vector<float> output(OUTPUT_DIM);

    // Trade state
    bool has_position = false;
    float entry_price = 0.0f;
    int ticks_since_trade = 999; 
    int sell_signal_streak = 0; 

    std::cerr << "--- ENGINE ONLINE (High-Accuracy Mode) ---" << std::endl;

    while (std::cin.read(reinterpret_cast<char*>(input_buffer.data()), READ_DIM * sizeof(float))) {
        
        float current_price = input_buffer[160];
        
        std::fill(hidden.begin(), hidden.end(), 0.0f);
        
        // Layer 1
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < INPUT_DIM; ++j) {
                hidden[i] += input_buffer[j] * model.w1[i * INPUT_DIM + j];
            }
            hidden[i] += model.b1[i];
        }
        relu(hidden);
        
        // Layer 2
        std::fill(output.begin(), output.end(), 0.0f);
        for (int i = 0; i < OUTPUT_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                output[i] += hidden[j] * model.w2[i * HIDDEN_DIM + j];
            }
            output[i] += model.b2[i];
        }
        softmax(output);

        float p_down = output[0];
        // float p_flat = output[1];
        float p_up   = output[2];
        
        std::cout << "TICK | " << current_price << " | " << p_up << " | " << p_down << std::endl;

        ticks_since_trade++;

        if (!has_position) {
            
            if (ticks_since_trade < COOLDOWN_TICKS) {
                 if (ticks_since_trade % 60 == 0) {
                     std::cerr << " [WAITING] Cooling down..." << std::endl;
                 }
                 continue;
            }

            // Buy
            if (p_up > CONFIDENCE_THRESHOLD) {
                std::cout << "ACTION | BUY | " << current_price << " | " << p_up << std::endl;
                has_position = true;
                entry_price = current_price;
                ticks_since_trade = 0;
                sell_signal_streak = 0;
            }
        } 
        else {
            // Sell
            float pnl = (current_price - entry_price) / entry_price;

            // Time Stop 
            if (ticks_since_trade > 300 && pnl < -0.0005) { 
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [TIME STOP] Stale trade. Exiting." << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                COOLDOWN_TICKS = 300;
                continue;
            }

            // Stop loss
            if (pnl < -STOP_LOSS) {
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [STOP LOSS] " << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                COOLDOWN_TICKS = 300; // PENALTY: Wait 5 mins (Stop Revenge Trading)
                continue;
            }

            // Take profit
            if (pnl > TAKE_PROFIT) {
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [TAKE PROFIT] " << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                COOLDOWN_TICKS = 10;
                continue;
            }

            // AI reversal
            if (p_down > 0.65) sell_signal_streak++;
            else sell_signal_streak = 0;

            if (sell_signal_streak >= CONFIRMATION_STREAK) {
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [AI EXIT] " << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                COOLDOWN_TICKS = (pnl > 0) ? 10 : 300; 
            }
        }
    }
    return 0;
}