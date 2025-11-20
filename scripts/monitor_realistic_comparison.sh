#!/bin/bash

# Monitor both DDPG (original) and PPO V2 (improved) training

while true; do
    clear
    echo "============================================================"
    echo "Realistic Portfolio Training - Comparison"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
    
    # DDPG Original
    echo "🔄 DDPG + Options (Original Config - risk_penalty=4.0)"
    if ps aux | grep -E "train_with_options.*ddpg.*config_realistic.yaml" | grep -v grep > /dev/null; then
        echo "   Status: ✓ Running"
        if [ -f "logs/ddpg_realistic_options_training.log" ]; then
            STEPS=$(grep -o "Steps: [0-9]*" logs/ddpg_realistic_options_training.log | tail -1 | awk '{print $2}')
            REWARD=$(grep -o "Mean Reward: [-.0-9]*" logs/ddpg_realistic_options_training.log | tail -1 | awk '{print $3}')
            if [ ! -z "$STEPS" ]; then
                PERCENT=$((STEPS * 100 / 750000))
                echo "   Progress: $STEPS / 750,000 (${PERCENT}%)"
                echo "   Mean Reward: $REWARD"
                printf "   ["
                FILLED=$((PERCENT / 2))
                for i in $(seq 1 50); do
                    if [ $i -le $FILLED ]; then printf "="; else printf " "; fi
                done
                printf "] ${PERCENT}%%\n"
            fi
        fi
    else
        echo "   Status: ✗ Not running / Completed"
    fi
    echo ""
    
    # PPO V2 Improved
    echo "🔄 PPO + Options (V2 Improved - risk_penalty=1.5, zero weights, optimized options)"
    if ps aux | grep -E "train_with_options.*ppo.*config_realistic_v2.yaml" | grep -v grep > /dev/null; then
        echo "   Status: ✓ Running"
        if [ -f "logs/ppo_realistic_v2_training.log" ]; then
            STEPS=$(grep -o "Steps: [0-9]*" logs/ppo_realistic_v2_training.log | tail -1 | awk '{print $2}')
            REWARD=$(grep -o "Mean Reward: [-.0-9]*" logs/ppo_realistic_v2_training.log | tail -1 | awk '{print $3}')
            if [ ! -z "$STEPS" ]; then
                PERCENT=$((STEPS * 100 / 750000))
                echo "   Progress: $STEPS / 750,000 (${PERCENT}%)"
                echo "   Mean Reward: $REWARD"
                printf "   ["
                FILLED=$((PERCENT / 2))
                for i in $(seq 1 50); do
                    if [ $i -le $FILLED ]; then printf "="; else printf " "; fi
                done
                printf "] ${PERCENT}%%\n"
            fi
        fi
    else
        echo "   Status: ✗ Not running / Completed"
    fi
    echo ""
    
    echo "============================================================"
    echo "Key Differences:"
    echo "  DDPG Original: risk_penalty=4.0, 100% options, no zero weights"
    echo "  PPO V2:        risk_penalty=1.5, 50% puts/30% calls, zero weights OK"
    echo ""
    echo "Expected: PPO V2 should achieve higher returns with similar drawdown"
    echo "============================================================"
    echo ""
    echo "Refreshing in 15 seconds... (Ctrl+C to exit)"
    
    sleep 15
done
