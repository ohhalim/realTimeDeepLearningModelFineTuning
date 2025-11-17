#!/bin/bash
# Complete Verification & Execution Script
# Run this to verify everything works

set -e  # Exit on error

echo "======================================================================"
echo "Practical Model - Complete Verification & Setup"
echo "======================================================================"

# Step 1: Check Python
echo -e "\n[1/6] Checking Python installation..."
python3 --version || {
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
}
echo "✓ Python found"

# Step 2: Create virtual environment
echo -e "\n[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate

# Step 3: Install dependencies
echo -e "\n[3/6] Installing dependencies..."
pip install --upgrade pip --quiet
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
pip install numpy --quiet
echo "✓ Dependencies installed"

# Step 4: Verify model
echo -e "\n[4/6] Testing model component..."
python simple_model.py || {
    echo "❌ Model test failed"
    exit 1
}
echo "✓ Model component works"

# Step 5: Verify tokenizer
echo -e "\n[5/6] Testing tokenizer component..."
python simple_tokenizer.py || {
    echo "❌ Tokenizer test failed"
    exit 1
}
echo "✓ Tokenizer component works"

# Step 6: Quick training test
echo -e "\n[6/6] Running quick training test (5 epochs)..."
python train_simple.py \
    --model_size tiny \
    --epochs 5 \
    --num_train 100 \
    --num_val 20 \
    --batch_size 4 \
    --output_dir ./test_outputs \
    || {
    echo "❌ Training test failed"
    exit 1
}
echo "✓ Training works"

echo -e "\n======================================================================"
echo "✅ ALL VERIFICATION PASSED!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Source the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Train with more epochs:"
echo "     python train_simple.py --epochs 50"
echo ""
echo "  3. Generate samples:"
echo "     python generate.py --checkpoint test_outputs/best_model.pt"
echo ""
echo "Model saved to: ./test_outputs/best_model.pt"
