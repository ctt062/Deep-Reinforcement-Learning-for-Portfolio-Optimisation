#!/bin/bash
# Train all three models: DDPG, PPO, DQN on 2010-2018, test on 2018-2020

echo "============================================================"
echo "  Final Benchmark Training: 2010-2018 → Test 2018-2020"
echo "============================================================"
echo ""
echo "Training Period: 2010-2018 (8 years)"
echo "Test Period: 2018-2020 (2 years including COVID crash)"
echo ""
echo "Models: DDPG V2, PPO V1, DQN"
echo "============================================================"
echo ""

# Create directories
mkdir -p models
mkdir -p results
mkdir -p logs

# Start training all three models in parallel
echo "Starting DDPG training..."
nohup python scripts/train_with_options.py \
  --agent ddpg \
  --config configs/config_final_benchmark.yaml \
  --output-dir models \
  > logs/train_ddpg_final.log 2>&1 &
DDPG_PID=$!
echo "✓ DDPG training started (PID: $DDPG_PID)"

sleep 2

echo "Starting PPO training..."
nohup python scripts/train_with_options.py \
  --agent ppo \
  --config configs/config_final_benchmark.yaml \
  --output-dir models \
  > logs/train_ppo_final.log 2>&1 &
PPO_PID=$!
echo "✓ PPO training started (PID: $PPO_PID)"

sleep 2

echo "Starting DQN training..."
nohup python scripts/train_with_options.py \
  --agent dqn \
  --config configs/config_final_benchmark.yaml \
  --output-dir models \
  > logs/train_dqn_final.log 2>&1 &
DQN_PID=$!
echo "✓ DQN training started (PID: $DQN_PID)"

echo ""
echo "============================================================"
echo "All models training in background"
echo "============================================================"
echo ""
echo "Process IDs:"
echo "  DDPG: $DDPG_PID"
echo "  PPO: $PPO_PID"
echo "  DQN: $DQN_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/train_ddpg_final.log"
echo "  tail -f logs/train_ppo_final.log"
echo "  tail -f logs/train_dqn_final.log"
echo ""
echo "Check status:"
echo "  ps aux | grep train_with_options"
echo ""
echo "Estimated completion: ~30-40 minutes"
echo "============================================================"
