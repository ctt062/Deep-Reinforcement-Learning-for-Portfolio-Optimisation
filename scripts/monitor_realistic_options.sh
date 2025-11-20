#!/bin/bash

# Continuous monitoring for PPO+Options and DDPG+Options training on realistic config
# Target: Sharpe > 1.0, Drawdown < 10%

while true; do
    clear
    echo "========================================"
    echo "Realistic Portfolio + Options Training"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    echo "Configuration:"
    echo "  - 33 Assets (Multi-sector, No Crypto)"
    echo "  - Weekly Frequency (2010-2024)"
    echo "  - Options Overlay (Puts + Calls)"
    echo "  - Risk Penalty: 4.0 (Very Aggressive)"
    echo "  - Target: Sharpe > 1.0, DD < 10%"
    echo ""
    echo "========================================"
    
    # Check PPO
    echo "🔄 PPO + Options Training:"
    if ps aux | grep -E "train_with_options.*ppo" | grep -v grep > /dev/null; then
        echo "   Status: ✓ Running"
        
        if [ -f "logs/ppo_realistic_options_training.log" ]; then
            PPO_STEPS=$(grep -o "Steps: [0-9]*" logs/ppo_realistic_options_training.log | tail -1 | awk '{print $2}')
            PPO_REWARD=$(grep -o "Mean Reward: [-.0-9]*" logs/ppo_realistic_options_training.log | tail -1 | awk '{print $3}')
            
            if [ ! -z "$PPO_STEPS" ]; then
                PPO_PERCENT=$((PPO_STEPS * 100 / 750000))
                echo "   Progress: $PPO_STEPS / 750,000 (${PPO_PERCENT}%)"
                echo "   Mean Reward: $PPO_REWARD"
                
                # Progress bar
                FILLED=$((PPO_PERCENT / 2))
                printf "   ["
                for i in $(seq 1 50); do
                    if [ $i -le $FILLED ]; then
                        printf "="
                    else
                        printf " "
                    fi
                done
                printf "] ${PPO_PERCENT}%%\n"
            else
                echo "   Status: Initializing..."
            fi
        fi
    else
        if [ -f "models/ppo_*_realistic_options_final.zip" ]; then
            echo "   Status: ✓ COMPLETED"
        else
            echo "   Status: ✗ Not running"
        fi
    fi
    echo ""
    
    # Check DDPG
    echo "🔄 DDPG + Options Training:"
    if ps aux | grep -E "train_with_options.*ddpg" | grep -v grep > /dev/null; then
        echo "   Status: ✓ Running"
        
        if [ -f "logs/ddpg_realistic_options_training.log" ]; then
            DDPG_STEPS=$(grep -o "Steps: [0-9]*" logs/ddpg_realistic_options_training.log | tail -1 | awk '{print $2}')
            DDPG_REWARD=$(grep -o "Mean Reward: [-.0-9]*" logs/ddpg_realistic_options_training.log | tail -1 | awk '{print $3}')
            
            if [ ! -z "$DDPG_STEPS" ]; then
                DDPG_PERCENT=$((DDPG_STEPS * 100 / 750000))
                echo "   Progress: $DDPG_STEPS / 750,000 (${DDPG_PERCENT}%)"
                echo "   Mean Reward: $DDPG_REWARD"
                
                # Progress bar
                FILLED=$((DDPG_PERCENT / 2))
                printf "   ["
                for i in $(seq 1 50); do
                    if [ $i -le $FILLED ]; then
                        printf "="
                    else
                        printf " "
                    fi
                done
                printf "] ${DDPG_PERCENT}%%\n"
            else
                echo "   Status: Initializing..."
            fi
        fi
    else
        if [ -f "models/ddpg_*_realistic_options_final.zip" ]; then
            echo "   Status: ✓ COMPLETED"
        else
            echo "   Status: ✗ Not running"
        fi
    fi
    echo ""
    
    # Check if both are done
    PPO_DONE=false
    DDPG_DONE=false
    
    if ! ps aux | grep -E "train_with_options.*ppo" | grep -v grep > /dev/null; then
        if ls models/*ppo*realistic*options*final.zip 1> /dev/null 2>&1; then
            PPO_DONE=true
        fi
    fi
    
    if ! ps aux | grep -E "train_with_options.*ddpg" | grep -v grep > /dev/null; then
        if ls models/*ddpg*realistic*options*final.zip 1> /dev/null 2>&1; then
            DDPG_DONE=true
        fi
    fi
    
    if [ "$PPO_DONE" = true ] && [ "$DDPG_DONE" = true ]; then
        echo "========================================"
        echo "🎉 ALL TRAINING COMPLETED! 🎉"
        echo "========================================"
        echo ""
        echo "Next steps:"
        echo "  1. Evaluate models:"
        echo "     python scripts/evaluate_with_options.py --agent ppo --model models/ppo_*_realistic_options_final.zip --config configs/config_realistic.yaml"
        echo "     python scripts/evaluate_with_options.py --agent ddpg --model models/ddpg_*_realistic_options_final.zip --config configs/config_realistic.yaml"
        echo ""
        break
    fi
    
    echo "========================================"
    echo "Refreshing in 15 seconds..."
    echo "Press Ctrl+C to exit"
    echo "========================================"
    
    sleep 15
done
