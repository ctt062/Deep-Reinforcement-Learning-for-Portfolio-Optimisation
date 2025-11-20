#!/bin/bash

# Continuous training monitor for V2
# Press Ctrl+C to stop

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         Training Status Monitor - V2 (Auto-refresh)           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "⏰ Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check running processes
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 PROCESS STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    PPO_PID=$(ps aux | grep "train.py.*ppo.*config_diversified_v2" | grep -v grep | awk '{print $2}')
    DDPG_PID=$(ps aux | grep "train.py.*ddpg.*config_diversified_v2" | grep -v grep | awk '{print $2}')
    DQN_PID=$(ps aux | grep "train.py.*dqn.*config_diversified_v2" | grep -v grep | awk '{print $2}')
    
    if [ -n "$PPO_PID" ]; then
        echo "✅ PPO:  Running (PID: $PPO_PID)"
    else
        echo "❌ PPO:  Not running"
    fi
    
    if [ -n "$DDPG_PID" ]; then
        echo "✅ DDPG: Running (PID: $DDPG_PID)"
    else
        echo "❌ DDPG: Not running"
    fi
    
    if [ -n "$DQN_PID" ]; then
        echo "✅ DQN:  Running (PID: $DQN_PID)"
    else
        echo "❌ DQN:  Not running"
    fi
    
    echo ""
    
    # PPO Progress
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔵 PPO TRAINING PROGRESS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -f logs/ppo_v2_training.log ]; then
        # Get latest timesteps
        LATEST=$(tail -20 logs/ppo_v2_training.log | grep -o "total_timesteps=[0-9]*" | tail -1)
        if [ -n "$LATEST" ]; then
            TIMESTEPS=$(echo $LATEST | grep -o "[0-9]*")
            PERCENT=$((TIMESTEPS * 100 / 500000))
            echo "Timesteps: $TIMESTEPS / 500,000 ($PERCENT%)"
        fi
        
        # Get latest reward
        REWARD=$(tail -20 logs/ppo_v2_training.log | grep "mean_reward" | tail -1)
        if [ -n "$REWARD" ]; then
            echo "$REWARD"
        fi
        
        # Check if completed
        if grep -q "Saving final model" logs/ppo_v2_training.log; then
            echo "✅ TRAINING COMPLETED!"
        fi
    else
        echo "⏳ Log file not created yet..."
    fi
    
    echo ""
    
    # DDPG Progress
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🟢 DDPG TRAINING PROGRESS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -f logs/ddpg_v2_training.log ]; then
        # Get latest timesteps
        LATEST=$(tail -20 logs/ddpg_v2_training.log | grep -o "total_timesteps=[0-9]*" | tail -1)
        if [ -n "$LATEST" ]; then
            TIMESTEPS=$(echo $LATEST | grep -o "[0-9]*")
            PERCENT=$((TIMESTEPS * 100 / 500000))
            echo "Timesteps: $TIMESTEPS / 500,000 ($PERCENT%)"
        fi
        
        # Get latest reward
        REWARD=$(tail -20 logs/ddpg_v2_training.log | grep "mean_reward" | tail -1)
        if [ -n "$REWARD" ]; then
            echo "$REWARD"
        fi
        
        # Check if completed
        if grep -q "Saving final model" logs/ddpg_v2_training.log; then
            echo "✅ TRAINING COMPLETED!"
        fi
    else
        echo "⏳ Log file not created yet..."
    fi
    
    echo ""
    
    # DQN Progress
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🟣 DQN TRAINING PROGRESS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -f logs/dqn_v2_training.log ]; then
        # Get latest timesteps
        LATEST=$(tail -20 logs/dqn_v2_training.log | grep -o "total_timesteps=[0-9]*" | tail -1)
        if [ -n "$LATEST" ]; then
            TIMESTEPS=$(echo $LATEST | grep -o "[0-9]*")
            PERCENT=$((TIMESTEPS * 100 / 500000))
            echo "Timesteps: $TIMESTEPS / 500,000 ($PERCENT%)"
        fi
        
        # Get latest reward
        REWARD=$(tail -20 logs/dqn_v2_training.log | grep "mean_reward" | tail -1)
        if [ -n "$REWARD" ]; then
            echo "$REWARD"
        fi
        
        # Check if completed
        if grep -q "Saving final model" logs/dqn_v2_training.log; then
            echo "✅ TRAINING COMPLETED!"
        fi
    else
        echo "⏳ Log file not created yet..."
    fi
    
    echo ""
    
    # Model files
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💾 COMPLETED MODELS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    MODEL_COUNT=$(ls models_diversified_v2/*.zip 2>/dev/null | wc -l | tr -d ' ')
    echo "Saved models: $MODEL_COUNT / 3"
    
    if [ -d models_diversified_v2 ]; then
        ls -lh models_diversified_v2/*.zip 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Auto-refreshing every 30 seconds... (Press Ctrl+C to stop)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if all done
    if [ "$MODEL_COUNT" = "3" ] && [ -z "$PPO_PID" ] && [ -z "$DDPG_PID" ] && [ -z "$DQN_PID" ]; then
        echo ""
        echo "🎉 ALL TRAINING COMPLETED! 🎉"
        echo ""
        echo "Run evaluation with:"
        echo "  bash scripts/evaluate_when_ready_v2.sh"
        break
    fi
    
    sleep 30
done
