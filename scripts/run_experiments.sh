#!/bin/bash
# Script to run the comprehensive experiments and analysis

echo "Starting comprehensive experiments..."
echo "This will run 60 experiments (6 forms × 5 permutations × 2 label types)"
echo "Estimated time: Several hours depending on your hardware and model size"
echo ""
echo "Press Ctrl+C to cancel or any key to continue..."
read -n 1 -s

# Create a log directory
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/experiment_run_$TIMESTAMP.log"

echo "Running experiments..."
echo "Log file: $LOG_FILE"

# Run the experiments with output logged
python ../run_comprehensive_experiments.py 2>&1 | tee $LOG_FILE

# Check if the experiments completed successfully
if [ $? -eq 0 ]; then
    echo "Experiments completed successfully!"
    echo "Running analysis..."
    
    # Run the analysis script
    python analysis/analyze_comprehensive_results.py
    
    echo "All done! Check the results directory for outputs."
else
    echo "Error: Experiments did not complete successfully."
    echo "Check the log file for details: $LOG_FILE"
fi
