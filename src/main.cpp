#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>

// CONFIGURATION
const float FEES = 0.0000; 
const float MIN_NET_PROFIT = 0.0002; 
const float STOP_LOSS = 0.0010; 
const float CONFIDENCE_THRESHOLD = 0.65; 
const int TICKS_PER_SEC = 50;
const int MAX_HOLD_TICKS = 10 * TICKS_PER_SEC; 
int PENALTY_COOLDOWN = 2 * TICKS_PER_SEC; 
int PROFIT_COOLDOWN = 1 * TICKS_PER_SEC;

const int INPUT_DIM = 160;   
const int READ_DIM = 161;    
const int HIDDEN1_DIM = 64;
const int HIDDEN2_DIM = 32;
const int OUTPUT_DIM = 3;

struct MarketModel {
    std::vector<float> w1; std::vector<float> b1;
    std::vector<float> w2; std::vector<float> b2;
    std::vector<float> w3; std::vector<float> b3;
};

void load_layer(std::ifstream& file, std::vector<float>& w, std::vector<float>& b, int rows, int cols) {
    w.resize(rows * cols);
    b.resize(rows);
    file.read(reinterpret_cast<char*>(w.data()), w.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(float));
}

MarketModel load_weights(const std::string& path) {
    MarketModel model;
    std::ifstream file(path, std::ios::binary);
    if (!file) { 
        std::cerr << "Error: No model weights found at " << path << std::endl; 
        exit(1); 
    }
    load_layer(file, model.w1, model.b1, HIDDEN1_DIM, INPUT_DIM);
    load_layer(file, model.w2, model.b2, HIDDEN2_DIM, HIDDEN1_DIM);
    load_layer(file, model.w3, model.b3, OUTPUT_DIM, HIDDEN2_DIM);
    return model;
}

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
    std::vector<float> hidden1(HIDDEN1_DIM);
    std::vector<float> hidden2(HIDDEN2_DIM);
    std::vector<float> output(OUTPUT_DIM);

    // Trading State
    bool has_position = false;
    float entry_price = 0.0f;
    int ticks_since_trade = 0; 
    int sell_signal_streak = 0; 
    int required_cooldown = 0;

    std::cerr << "--- HFT SCALPER ONLINE ---" << std::endl;
    std::cerr << "   - Target: " << (MIN_NET_PROFIT * 100) << "%" << std::endl;

    while (std::cin.read(reinterpret_cast<char*>(input_buffer.data()), READ_DIM * sizeof(float))) {
        
        float current_price = input_buffer[160];
        
        // Inference: fc1 -> relu -> fc2 -> relu -> fc3 -> softmax
        std::fill(hidden1.begin(), hidden1.end(), 0.0f);
        for (int i = 0; i < HIDDEN1_DIM; ++i) {
            for (int j = 0; j < INPUT_DIM; ++j)
                hidden1[i] += input_buffer[j] * model.w1[i * INPUT_DIM + j];
            hidden1[i] += model.b1[i];
        }
        relu(hidden1);

        std::fill(hidden2.begin(), hidden2.end(), 0.0f);
        for (int i = 0; i < HIDDEN2_DIM; ++i) {
            for (int j = 0; j < HIDDEN1_DIM; ++j)
                hidden2[i] += hidden1[j] * model.w2[i * HIDDEN1_DIM + j];
            hidden2[i] += model.b2[i];
        }
        relu(hidden2);

        std::fill(output.begin(), output.end(), 0.0f);
        for (int i = 0; i < OUTPUT_DIM; ++i) {
            for (int j = 0; j < HIDDEN2_DIM; ++j)
                output[i] += hidden2[j] * model.w3[i * HIDDEN2_DIM + j];
            output[i] += model.b3[i];
        }
        softmax(output);

        float p_down = output[0];
        float p_up   = output[2];
        
        // Feedback
        std::cout << "TICK | " << current_price << " | " << p_up << " | " << p_down << std::endl;

        ticks_since_trade++;

        if (!has_position) {
            
            if (ticks_since_trade < required_cooldown) {
                 if (ticks_since_trade % (TICKS_PER_SEC * 2) == 0) {
                     std::cerr << " [WAITING] Cooldown. " << std::endl;
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
                required_cooldown = 0;
            }
        } 
        else {
            // Sell
            
            // PnL
            float raw_pnl = (current_price - entry_price) / entry_price;

            // Time Stop
            if (ticks_since_trade > MAX_HOLD_TICKS && raw_pnl < 0) { 
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [TIME STOP] Too slow. Next!" << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                required_cooldown = PENALTY_COOLDOWN;
                continue;
            }

            // Stop Loss
            if (raw_pnl < -STOP_LOSS) {
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [STOP LOSS] Cut loss." << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                required_cooldown = PENALTY_COOLDOWN;
                continue;
            }

            // Take Profit
            if (raw_pnl > MIN_NET_PROFIT) {
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [SCALP] Bagged +" << (raw_pnl*100) << "%" << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                required_cooldown = PROFIT_COOLDOWN;
                continue;
            }

            if (p_down > 0.70) sell_signal_streak++;
            else sell_signal_streak = 0;

            if (sell_signal_streak >= 8) {
                std::cout << "ACTION | SELL | " << current_price << " | " << p_down << std::endl;
                std::cerr << " [AI EXIT] Dumping." << std::endl;
                has_position = false;
                ticks_since_trade = 0;
                required_cooldown = PENALTY_COOLDOWN; 
            }
        }
    }
    return 0;
}