#!/bin/bash
# Run all three MNIST methods on Gazelle (as root via sudo)
cd /home/uisrc/mnist
for m in dsq ste lsqplus; do
  echo "===== METHOD=$m ====="
  echo 5182 | sudo -S -p X timeout 900 env MNIST_METHOD=$m MNIST_LIMIT=10000 MNIST_BATCH=50 MNIST_MODE=scale \
    python3 run_mnist_gazelle.py 2>&1 | grep -E "MNIST method|FINAL|NumPy|gap"
done
echo "===== ALL DONE ====="
