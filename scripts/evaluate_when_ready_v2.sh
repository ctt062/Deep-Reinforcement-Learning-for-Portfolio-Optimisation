#!/bin/bash
# Evaluate all V2 models once training completes

echo "======================================================================"
echo "Evaluating Diversified Portfolio V2 Models"
echo "======================================================================"
echo ""

CONFIG="configs/config_diversified_v2.yaml"
RESULTS_DIR="results_diversified_v2"

# Wait for all training to complete
while true; do
    ppo_done=$(ls models_diversified_v2/ppo_*_final.zip 2>/dev/null | wc -l)
    ddpg_done=$(ls models_diversified_v2/ddpg_*_final.zip 2>/dev/null | wc -l)
    dqn_done=$(ls models_diversified_v2/dqn_*_final.zip 2>/dev/null | wc -l)
    
    total_done=$((ppo_done + ddpg_done + dqn_done))
    
    echo "Completed models: $total_done/3"
    
    if [ $total_done -eq 3 ]; then
        echo "All models training completed!"
        break
    fi
    
    echo "Waiting for training to complete... (checking again in 5 minutes)"
    sleep 300
done

echo ""
echo "======================================================================"
echo "Evaluating PPO Model"
echo "======================================================================"
PPO_MODEL=$(ls models_diversified_v2/ppo_*_final.zip | head -1)
python scripts/evaluate.py --agent ppo --model-path "$PPO_MODEL" --config "$CONFIG" --save-results

echo ""
echo "======================================================================"
echo "Evaluating DDPG Model"
echo "======================================================================"
DDPG_MODEL=$(ls models_diversified_v2/ddpg_*_final.zip | head -1)
python scripts/evaluate.py --agent ddpg --model-path "$DDPG_MODEL" --config "$CONFIG" --save-results

echo ""
echo "======================================================================"
echo "Evaluating DQN Model"
echo "======================================================================"
DQN_MODEL=$(ls models_diversified_v2/dqn_*_final.zip | head -1)
python scripts/evaluate.py --agent dqn --model-path "$DQN_MODEL" --config "$CONFIG" --save-results

echo ""
echo "======================================================================"
echo "Creating Comparison Summary"
echo "======================================================================"
python scripts/evaluate.py --compare-all --config "$CONFIG" --save-results

echo ""
echo "======================================================================"
echo "Evaluation Complete! Check $RESULTS_DIR for results."
echo "======================================================================"
