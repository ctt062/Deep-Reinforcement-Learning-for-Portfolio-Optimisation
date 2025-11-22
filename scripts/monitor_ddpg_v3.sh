#!/bin/bash
# Real-time monitoring for DDPG V3 - Final push to Sharpe > 1.0

clear

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "  DDPG V3 Training - FINAL OPTIMIZATION"
    echo "════════════════════════════════════════════════════════════"
    date
    echo ""
    echo "🎯 Goal: Push Sharpe from 0.988 → >1.0 while maintaining DD <10%"
    echo ""
    
    # Define target timesteps
    TARGET=500000
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DDPG V3: Final Fine-Tuning"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Changes from V2 → V3:"
    echo "  • Risk Penalty: 0.8 → 0.75 (more aggressive)"
    echo "  • Max Position: 25% → 27% (more concentration)"
    echo "  • Turnover Penalty: 0.0005 → 0.0003 (more trading)"
    echo "  • Covered Calls: 25% → 30% (more income)"
    echo "  • Call Strike: 104% → 103.5% (more aggressive)"
    echo "  • Learning Rate: 1e-4 → 1.2e-4 (faster)"
    echo ""
    
    if ps aux | grep -q "[t]rain_with_options.*ddpg.*ddpg_v3"; then
        echo "Status: ✓ RUNNING"
        echo ""
        
        if [ -f logs/train_ddpg_v3.log ]; then
            # Get latest timestep
            latest_ts=$(grep -oE "total_timesteps[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ddpg_v3.log | tail -1 | grep -oE "[0-9]+")
            
            if [ -n "$latest_ts" ]; then
                pct=$(awk "BEGIN {printf \"%.1f\", ($latest_ts/$TARGET)*100}")
                bar_length=$(awk "BEGIN {printf \"%.0f\", ($latest_ts/$TARGET)*50}")
                bar=$(printf "%-50s" "$(printf '#%.0s' $(seq 1 $bar_length))")
                
                echo "Progress:"
                echo "[$bar] $pct%"
                echo ""
                echo "Steps: $latest_ts / $TARGET"
                
                # Calculate remaining
                remaining=$((TARGET - latest_ts))
                if [ $latest_ts -gt 0 ]; then
                    time_elapsed=$(grep -oE "time_elapsed[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ddpg_v3.log | tail -1 | grep -oE "[0-9]+")
                    if [ -n "$time_elapsed" ] && [ $time_elapsed -gt 0 ]; then
                        rate=$(awk "BEGIN {printf \"%.0f\", $latest_ts/$time_elapsed}")
                        eta=$(awk "BEGIN {printf \"%.0f\", $remaining/$rate}")
                        echo "ETA: ~$((eta/60)) minutes"
                    fi
                fi
                
                echo ""
                
                # Get latest reward
                latest_reward=$(grep -oE "ep_rew_mean[[:space:]]*\|[[:space:]]*[0-9.e+]+" logs/train_ddpg_v3.log | tail -1 | awk '{print $NF}')
                if [ -n "$latest_reward" ]; then
                    echo "Latest Mean Reward: $latest_reward"
                fi
                
                # Get actor loss
                actor_loss=$(grep -oE "actor_loss[[:space:]]*\|[[:space:]]*-?[0-9.e+]+" logs/train_ddpg_v3.log | tail -1 | awk '{print $NF}')
                if [ -n "$actor_loss" ]; then
                    echo "Actor Loss: $actor_loss"
                fi
                
                echo ""
                
                # Time info
                time_elapsed=$(grep -oE "time_elapsed[[:space:]]*\|[[:space:]]*[0-9]+" logs/train_ddpg_v3.log | tail -1 | grep -oE "[0-9]+")
                if [ -n "$time_elapsed" ]; then
                    echo "Time Elapsed: ${time_elapsed}s (~$((time_elapsed/60)) min)"
                fi
            else
                echo "Progress: Starting up..."
                echo ""
                tail -5 logs/train_ddpg_v3.log | grep -E "Loading|Downloading|Building"
            fi
        else
            echo "Progress: Waiting for log file..."
        fi
    else
        echo "Status: ✗ NOT RUNNING"
        if [ -f logs/train_ddpg_v3.log ]; then
            echo ""
            echo "Training completed or stopped. Last log output:"
            tail -5 logs/train_ddpg_v3.log
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Results Comparison"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    printf "%-12s | %-10s | %-10s | %-10s\n" "Model" "Sharpe" "Drawdown" "Return"
    echo "-----------------------------------------------------------"
    printf "%-12s | %-10s | %-10s | %-10s\n" "DDPG V1" "0.9356 ✗" "9.50% ✓" "14.18% ✗"
    printf "%-12s | %-10s | %-10s | %-10s\n" "DDPG V2" "0.9881 🔥" "9.09% ✓" "14.34% ✗"
    printf "%-12s | %-10s | %-10s | %-10s\n" "DDPG V3" "??? 🎯" "??? 🎯" "??? 🎯"
    echo ""
    printf "%-12s | %-10s | %-10s | %-10s\n" "Target" ">1.0" "<10%" ">15%"
    echo ""
    echo "V2 was 98.8% there on Sharpe! V3 should push it over 1.0!"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Press Ctrl+C to stop monitoring | Auto-refresh: 10s"
    echo "════════════════════════════════════════════════════════════"
    
    sleep 10
done
