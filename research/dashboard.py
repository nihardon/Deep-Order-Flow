import sys
import time

INITIAL_BALANCE = 1000.0
FEE_RATE = 0.0005 

def main():
    usd = INITIAL_BALANCE
    btc = 0.0
    trades = 0
    in_position = False
    
    print(f"\n{'TIME':<10} | {'ACTION':<6} | {'PRICE':<10} | {'CONF':<5} | {'BALANCE':<10} | {'PnL':<8}")
    print("-" * 70)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            
            parts = line.split(" | ")
            msg_type = parts[0]
            
            if msg_type == "TICK":
                price = float(parts[1])
                conf_up = float(parts[2])
                conf_down = float(parts[3])
                
                # Calculate Live Value
                current_val = usd + (btc * price)
                pnl = ((current_val - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
                
                sys.stdout.write(f"\r LIVE: ${price:.2f} | PnL: {pnl:+.4f}% | Conf: {int(conf_up*100)}% (UP) vs {int(conf_down*100)}% (DOWN)   ")
                sys.stdout.flush()
                continue

            # Trades
            if msg_type == "ACTION":
                action = parts[1]
                price = float(parts[2])
                conf = float(parts[3])
                
                executed = False
                
                if action == "BUY" and usd > 10:
                    btc = (usd * (1 - FEE_RATE)) / price
                    usd = 0.0
                    trades += 1
                    in_position = True
                    executed = True
                    
                elif action == "SELL" and btc > 0.0001:
                    usd = (btc * price) * (1 - FEE_RATE)
                    btc = 0.0
                    trades += 1
                    in_position = False
                    executed = True

                if executed:
                    # Clear the live line
                    sys.stdout.write("\r" + " "*80 + "\r") 
                    
                    current_val = usd + (btc * price)
                    pnl = ((current_val - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
                    t_str = time.strftime("%H:%M:%S")
                    
                    print(f"{t_str:<10} | {action:<6} | {price:<10.2f} | {int(conf*100)}%  | ${current_val:<10.2f} | {pnl:+.2f}%")

    except KeyboardInterrupt:
        print("\n--- SESSION CLOSED ---")

if __name__ == "__main__":
    main()