#!/bin/bash
# Real-time monitoring for V3 optimization training

clear

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "  Training Monitor - V3 Optimizations (Real-Time)"
    echo "════════════════════════════════════════════════════════════"
    date
    echo ""
    
    # Define target timesteps
    TARGET=500000
    
    # Check PPO V3
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PPO V3: Risk Penalty 1.5 + Aggressive Stop-Loss"
    echo "  Target: Reduce DD from 18.5% to <10%, keep Sharpe >1.0"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if ps aux | grep -q "[t]rain_with_options.*ppo.*ppo_v3"; then
        echo "Status: ✓ RUNNING"
        
        if [ -f logs/train_ppo_v3.log ]; then
            # Get latest timestep
            latest_ts=$(grep -oE "total_timesteps[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ppo_v3.log | tail -1 | grep -oE "[0-9]+")
            
            if [ -n "$latest_ts" ]; then
                pct=$(awk "BEGIN {printf \"%.1f\", ($latest_ts/$TARGET)*100}")
                bar_length=$(awk "BEGIN {printf \"%.0f\", ($latest_ts/$TARGET)*50}")
                bar=$(printf "%-50s" "$(printf '#%.0s' $(seq 1 $bar_length))")
                
                echo "Progress: [$bar] $pct%"
                echo "Steps: $latest_ts / $TARGET"
                
                # Get latest reward
                latest_reward=$(grep -oE "ep_rew_mean[[:space:]]*\|[[:space:]]*[0-9.e+]+" logs/train_ppo_v3.log | tail -1 | awk '{print $NF}')
                if [ -n "$latest_reward" ]; then
                    echo "Latest Reward: $latest_reward"
                fi
                
                # Time info
                time_elapsed=$(grep -oE "time_elapsed[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ppo_v3.log | tail -1 | grep -oE "[0-9]+")
                if [ -n "$time_elapsed" ]; then
                    echo "Time Elapsed: ${time_elapsed}s (~$((time_elapsed/60)) min)"
                fi
            else
                echo "Progress: Starting up..."
            fi
        else
            echo "Progress: Waiting for log file..."
        fi
    else
        echo "Status: ✗ NOT RUNNING"
        if [ -f logs/train_ppo_v3.log ]; then
            echo "Last log output:"
            tail -3 logs/train_ppo_v3.log
        fi
    fi
    
    echo ""
    
    # Check DDPG V2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DDPG V2: Risk Penalty 0.8 + Optimized for Returns"
    echo "  Target: Push Sharpe from 0.94 to >1.0, keep DD <10%"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if ps aux | grep -q "[t]rain_with_options.*ddpg.*ddpg_v2"; then
        echo "Status: ✓ RUNNING"
        
        if [ -f logs/train_ddpg_v2.log ]; then
            latest_ts=$(grep -oE "total_timesteps[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ddpg_v2.log | tail -1 | grep -oE "[0-9]+")
            
            if [ -n "$latest_ts" ]; then
                pct=$(awk "BEGIN {printf \"%.1f\", ($latest_ts/$TARGET)*100}")
                bar_length=$(awk "BEGIN {printf \"%.0f\", ($latest_ts/$TARGET)*50}")
                bar=$(printf "%-50s" "$(printf '#%.0s' $(seq 1 $bar_length))")
                
                echo "Progress: [$bar] $pct%"
                echo "Steps: $latest_ts / $TARGET"
                
                latest_reward=$(grep -oE "ep_rew_mean[[:space:]]*\|[[:space:]]*[0-9.e+]+" logs/train_ddpg_v2.log | tail -1 | awk '{print $NF}')
                if [ -n "$latest_reward" ]; then
                    echo "Latest Reward: $latest_reward"
                fi
                
                time_elapsed=$(grep -oE "time_elapsed[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ddpg_v2.log | tail -1 | grep -oE "[0-9]+")
                if [ -n "$time_elapsed" ]; then
                    echo "Time Elapsed: ${time_elapsed}s (~$((time_elapsed/60)) min)"
                fi
            else
                echo "Progress: Starting up..."
            fi
        else
            echo "Progress: Waiting for log file..."
        fi
    else
        echo "Status: ✗ NOT RUNNING"
        if [ -f logs/train_ddpg_v2.log ]; then
            echo "Last log output:"
            tail -3 logs/train_ddpg_v2.log
        fi
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Previous Results (for comparison)"
    echo "════════════════════════════════════════════════════════════"
    echo "  PPO V1: Sharpe 1.06 ✓ | DD 18.5% ✗ | Return 15.9% ✓"
    echo "  DDPG V1: Sharpe 0.94 ✗ | DD 9.5% ✓ | Return 14.2% ✗"
    echo ""
    echo "  Target: Sharpe >1.0 AND DD <10% AND Return >15%"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done
