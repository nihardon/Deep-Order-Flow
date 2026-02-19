import sys
import time

# --- CONFIGURATION ---
INITIAL_BALANCE = 1000.0
FEE_RATE = 0.0000 

def main():
    usd = INITIAL_BALANCE
    btc = 0.0
    trades = 0
    wins = 0
    losses = 0
    entry_val = 0
    
    start_time = time.time()
    last_update_time = 0
    
    # ANSI Colors
    G, R, B, Y, C, W = "\033[92m", "\033[91m", "\033[94m", "\033[93m", "\033[96m", "\033[0m"

    print(f"\n{B}HFT MONITORING TERMINAL{W}")
    print(f"{'TIME':<10} | {'ACTION':<6} | {'PRICE':<10} | {'CONF':<5} | {'BALANCE':<10} | {'PnL':<8}")
    print("-" * 75)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            
            try:
                parts = line.split(" | ")
                msg_type = parts[0]
                
                if msg_type == "TICK":
                    if time.time() - last_update_time < 0.1: # Faster refresh for scalper
                        continue
                    
                    last_update_time = time.time()
                    price, conf_up, conf_down = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    current_val = usd + (btc * price)
                    pnl = ((current_val - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
                    
                    # Status logic
                    arrow = f"{G}↑{W}" if conf_up > conf_down else f"{R}↓{W}"
                    win_rate = (wins / trades * 100) if trades > 0 else 0
                    
                    # Live Status Line
                    sys.stdout.write(f"\r {B}[LIVE]{W} ${price:.2f} | PnL: {pnl:+.3f}% | WinRate: {win_rate:.1f}% | Model: {int(max(conf_up, conf_down)*100)}% {arrow}   ")
                    sys.stdout.flush()
                    continue

                if msg_type == "ACTION":
                    action, price, conf = parts[1], float(parts[2]), float(parts[3])
                    executed = False
                    
                    if action == "BUY" and usd > 10:
                        entry_val = usd
                        btc = (usd * (1 - FEE_RATE)) / price
                        usd = 0.0
                        trades += 1
                        executed = True
                        color = G
                    elif action == "SELL" and btc > 0.0001:
                        usd = (btc * price) * (1 - FEE_RATE)
                        
                        # Track Win/Loss for the scalp
                        if usd > entry_val: wins += 1
                        else: losses += 1
                        
                        btc = 0.0
                        trades += 1
                        executed = True
                        color = R

                    if executed:
                        sys.stdout.write("\r" + " "*80 + "\r") 
                        current_val = usd + (btc * price)
                        pnl = ((current_val - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
                        t_str = time.strftime("%H:%M:%S")
                        print(f"{t_str:<10} | {color}{action:<6}{W} | {price:<10.2f} | {int(conf*100)}%  | ${current_val:<10.2f} | {pnl:+.2f}%")

            except Exception: continue

    except KeyboardInterrupt:
        uptime = time.time() - start_time
        print(f"\n\n{B}--- SESSION PERFORMANCE SUMMARY ---{W}")
        print(f"Total Trades: {trades} | Win Rate: {(wins/trades*100 if trades > 0 else 0):.2f}%")
        print(f"Final Value:  ${usd + (btc * price if btc > 0 else 0):.2f}")
        print(f"System Uptime: {int(uptime)} seconds")

if __name__ == "__main__":
    main()