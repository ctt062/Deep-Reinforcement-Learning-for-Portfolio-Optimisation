#!/bin/bash

# Continuous monitoring for remaining training (DDPG and DQN)
# Press Ctrl+C to exit

echo "========================================"
echo "Continuous Training Monitor"
echo "Monitoring DDPG and DQN (PPO Complete!)"
echo "Press Ctrl+C to exit"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "Training Status - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # Check PPO (should be complete)
    echo "✅ PPO: COMPLETED (500,000 timesteps)"
    if [ -f "models/ppo_20251119_145007_final.zip" ]; then
        echo "   Model: models/ppo_20251119_145007_final.zip"
    fi
    echo ""
    
    # Check DDPG
    echo "🔄 DDPG Training:"
    if ps aux | grep -E "train.py.*ddpg" | grep -v grep > /dev/null; then
        echo "   Status: ✓ Running"
        
        if [ -f "logs/ddpg_v2_training.log" ]; then
            # Get latest timestep
            DDPG_STEPS=$(grep -o "Steps: [0-9]*" logs/ddpg_v2_training.log | tail -1 | awk '{print $2}')
            DDPG_REWARD=$(grep -o "Mean Reward: [-.0-9]*" logs/ddpg_v2_training.log | tail -1 | awk '{print $3}')
            
            if [ ! -z "$DDPG_STEPS" ]; then
                DDPG_PERCENT=$((DDPG_STEPS * 100 / 500000))
                echo "   Progress: $DDPG_STEPS / 500,000 (${DDPG_PERCENT}%)"
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
            fi
        fi
    else
        if [ -f "models/ddpg_20251119_145013_final.zip" ]; then
            echo "   Status: ✓ COMPLETED"
        else
            echo "   Status: ✗ Not running (may have crashed)"
        fi
    fi
    echo ""
    
    # Check DQN
    echo "🔄 DQN Training:"
    if ps aux | grep -E "train.py.*dqn" | grep -v grep > /dev/null; then
        echo "   Status: ✓ Running"
        
        if [ -f "logs/dqn_v2_training.log" ]; then
            # Get latest timestep
            DQN_STEPS=$(grep -o "Steps: [0-9]*" logs/dqn_v2_training.log | tail -1 | awk '{print $2}')
            DQN_REWARD=$(grep -o "Mean Reward: [-.0-9]*" logs/dqn_v2_training.log | tail -1 | awk '{print $3}')
            
            if [ ! -z "$DQN_STEPS" ]; then
                DQN_PERCENT=$((DQN_STEPS * 100 / 500000))
                echo "   Progress: $DQN_STEPS / 500,000 (${DQN_PERCENT}%)"
                echo "   Mean Reward: $DQN_REWARD"
                
                # Progress bar
                FILLED=$((DQN_PERCENT / 2))
                printf "   ["
                for i in $(seq 1 50); do
                    if [ $i -le $FILLED ]; then
                        printf "="
                    else
                        printf " "
                    fi
                done
                printf "] ${DQN_PERCENT}%%\n"
            fi
        fi
    else
        if [ -f "models/dqn_20251119_145021_final.zip" ]; then
            echo "   Status: ✓ COMPLETED"
        else
            echo "   Status: ✗ Not running (may have crashed)"
        fi
    fi
    echo ""
    
    # Check if all are done
    ALL_DONE=true
    if ! [ -f "models/ddpg_20251119_145013_final.zip" ]; then
        ALL_DONE=false
    fi
    if ! [ -f "models/dqn_20251119_145021_final.zip" ]; then
        ALL_DONE=false
    fi
    
    if [ "$ALL_DONE" = true ]; then
        echo "========================================"
        echo "🎉 ALL TRAINING COMPLETED! 🎉"
        echo "========================================"
        echo ""
        echo "Final models:"
        echo "  - models/ppo_20251119_145007_final.zip"
        echo "  - models/ddpg_20251119_145013_final.zip"
        echo "  - models/dqn_20251119_145021_final.zip"
        echo ""
        echo "Run evaluation:"
        echo "  bash scripts/evaluate_when_ready_v2.sh"
        echo ""
        break
    fi
    
    echo "========================================"
    echo "Refreshing in 10 seconds..."
    echo "Press Ctrl+C to exit"
    echo "========================================"
    
    sleep 10
done
