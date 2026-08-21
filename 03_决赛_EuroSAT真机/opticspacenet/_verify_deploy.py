import sys
sys.path.insert(0, 'E:/LT-Simulator/train-test/src/core')
sys.path.insert(0, 'E:/LT-Simulator/train-test/src')
sys.path.insert(0, '.')
from gazelle_engine import MODEL_REGISTRY, build_model, NumpyBackend
import torch

m5, e5 = build_model(None, NumpyBackend(), model_name='model5')
m6, e6 = build_model(None, NumpyBackend(), model_name='model6')
x = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    print('M5 out:', tuple(m5(x).shape), '| M6 out:', tuple(m6(x).shape))
print('M5/M6 deployment classes OK')
