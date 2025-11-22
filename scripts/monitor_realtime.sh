#!/bin/bash
# Real-time monitoring for daily frequency training with live progress updates

TOTAL_TIMESTEPS=500000

while true; do
    clear
    echo "============================================================"
    echo "REAL-TIME Training Monitor - Daily Frequency + Stop-Loss"
    echo "============================================================"
    echo "Configuration: 18 assets, daily frequency, tiered stop-loss"
    echo "Target: Sharpe > 1.0, Max Drawdown < 10%"
    echo "Total timesteps per agent: 500,000"
    echo "============================================================"
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check if processes are running
    echo "=== PROCESS STATUS ==="
    ppo_proc=$(ps aux | grep "[t]rain_with_options.*ppo.*daily_stoploss" | wc -l | tr -d ' ')
    ddpg_proc=$(ps aux | grep "[t]rain_with_options.*ddpg.*daily_stoploss" | wc -l | tr -d ' ')

    if [ $ppo_proc -gt 0 ]; then
        ppo_cpu=$(ps aux | grep "[t]rain_with_options.*ppo.*daily_stoploss" | awk '{print $3}' | head -1)
        ppo_mem=$(ps aux | grep "[t]rain_with_options.*ppo.*daily_stoploss" | awk '{print $4}' | head -1)
        echo "✓ PPO:  RUNNING  | CPU: ${ppo_cpu}% | Memory: ${ppo_mem}%"
    else
        echo "✗ PPO:  STOPPED"
    fi

    if [ $ddpg_proc -gt 0 ]; then
        ddpg_cpu=$(ps aux | grep "[t]rain_with_options.*ddpg.*daily_stoploss" | awk '{print $3}' | head -1)
        ddpg_mem=$(ps aux | grep "[t]rain_with_options.*ddpg.*daily_stoploss" | awk '{print $4}' | head -1)
        echo "✓ DDPG: RUNNING  | CPU: ${ddpg_cpu}% | Memory: ${ddpg_mem}%"
    else
        echo "✗ DDPG: STOPPED"
    fi

    echo ""
    echo "=== PPO TRAINING PROGRESS ==="
    if [ -f logs/train_ppo_daily_stoploss.log ]; then
        # Get latest timesteps
        ppo_steps=$(grep "total_timesteps" logs/train_ppo_daily_stoploss.log | tail -1 | grep -oE '[0-9]+' | tail -1)
        
        if [ -n "$ppo_steps" ] && [ "$ppo_steps" -gt 0 ]; then
            ppo_percent=$(awk "BEGIN {printf \"%.1f\", ($ppo_steps / $TOTAL_TIMESTEPS) * 100}")
            ppo_bar_length=$(awk "BEGIN {printf \"%.0f\", ($ppo_steps / $TOTAL_TIMESTEPS) * 50}")
            ppo_bar=$(printf '%*s' "$ppo_bar_length" | tr ' ' '█')
            ppo_empty=$(printf '%*s' "$((50 - ppo_bar_length))" | tr ' ' '░')
            
            echo "Progress: [${ppo_bar}${ppo_empty}] ${ppo_percent}%"
            echo "Steps: ${ppo_steps} / ${TOTAL_TIMESTEPS}"
            
            # Get latest reward
            ppo_reward=$(grep "ep_rew_mean" logs/train_ppo_daily_stoploss.log | tail -1 | grep -oE '[0-9.]+e\+[0-9]+' | head -1)
            if [ -n "$ppo_reward" ]; then
                echo "Mean Reward: ${ppo_reward}"
            fi
            
            # Estimate time remaining
            time_elapsed=$(grep "time_elapsed" logs/train_ppo_daily_stoploss.log | tail -1 | grep -oE '[0-9]+' | tail -1)
            if [ -n "$time_elapsed" ] && [ "$time_elapsed" -gt 0 ] && [ "$ppo_steps" -gt 0 ]; then
                steps_per_sec=$(awk "BEGIN {printf \"%.1f\", $ppo_steps / $time_elapsed}")
                remaining_steps=$((TOTAL_TIMESTEPS - ppo_steps))
                eta_seconds=$(awk "BEGIN {printf \"%.0f\", $remaining_steps / $steps_per_sec}")
                eta_minutes=$((eta_seconds / 60))
                echo "Speed: ${steps_per_sec} steps/sec | ETA: ${eta_minutes} minutes"
            fi
        else
            echo "Status: Initializing..."
        fi
    else
        echo "Status: No log file yet"
    fi

    echo ""
    echo "=== DDPG TRAINING PROGRESS ==="
    if [ -f logs/train_ddpg_daily_stoploss.log ]; then
        # Get latest timesteps
        ddpg_steps=$(grep "total_timesteps" logs/train_ddpg_daily_stoploss.log | tail -1 | grep -oE '[0-9]+' | tail -1)
        
        if [ -n "$ddpg_steps" ] && [ "$ddpg_steps" -gt 0 ]; then
            ddpg_percent=$(awk "BEGIN {printf \"%.1f\", ($ddpg_steps / $TOTAL_TIMESTEPS) * 100}")
            ddpg_bar_length=$(awk "BEGIN {printf \"%.0f\", ($ddpg_steps / $TOTAL_TIMESTEPS) * 50}")
            ddpg_bar=$(printf '%*s' "$ddpg_bar_length" | tr ' ' '█')
            ddpg_empty=$(printf '%*s' "$((50 - ddpg_bar_length))" | tr ' ' '░')
            
            echo "Progress: [${ddpg_bar}${ddpg_empty}] ${ddpg_percent}%"
            echo "Steps: ${ddpg_steps} / ${TOTAL_TIMESTEPS}"
            
            # Get latest reward
            ddpg_reward=$(grep "ep_rew_mean" logs/train_ddpg_daily_stoploss.log | tail -1 | grep -oE '[0-9.]+e\+[0-9]+' | head -1)
            if [ -n "$ddpg_reward" ]; then
                echo "Mean Reward: ${ddpg_reward}"
            fi
            
            # Estimate time remaining
            time_elapsed=$(grep "time_elapsed" logs/train_ddpg_daily_stoploss.log | tail -1 | grep -oE '[0-9]+' | tail -1)
            if [ -n "$time_elapsed" ] && [ "$time_elapsed" -gt 0 ] && [ "$ddpg_steps" -gt 0 ]; then
                steps_per_sec=$(awk "BEGIN {printf \"%.1f\", $ddpg_steps / $time_elapsed}")
                remaining_steps=$((TOTAL_TIMESTEPS - ddpg_steps))
                eta_seconds=$(awk "BEGIN {printf \"%.0f\", $remaining_steps / $steps_per_sec}")
                eta_minutes=$((eta_seconds / 60))
                echo "Speed: ${steps_per_sec} steps/sec | ETA: ${eta_minutes} minutes"
            fi
        else
            echo "Status: Initializing..."
        fi
    else
        echo "Status: No log file yet"
    fi

    echo ""
    echo "=== SAVED MODELS ==="
    if [ -d models_daily_stoploss ]; then
        model_count=$(ls -1 models_daily_stoploss/*.zip 2>/dev/null | wc -l | tr -d ' ')
        echo "Model checkpoints: ${model_count}"
        if [ "$model_count" -gt 0 ]; then
            echo "Latest models:"
            ls -lht models_daily_stoploss/*.zip 2>/dev/null | head -3 | awk '{print "  " $9 " (" $5 ")"}'
        fi
    else
        echo "No models directory yet"
    fi

    echo ""
    echo "============================================================"
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing every 5 seconds..."
    echo "============================================================"
    
    sleep 5
done
