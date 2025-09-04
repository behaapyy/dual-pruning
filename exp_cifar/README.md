
---
## 🚀 Usage  
- **`exp_cifar`** for CIFAR (10, 100) experiments  

---
## Train Classifiers on the Entire Dataset
```javascript
python train.py --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 100 --dynamics --save_path ./checkpoint/all-dataset
```

## Sample Importance Evaluation
```javascript
python importance_evaluation.py --dynamics_path ./checkpoint/all-dataset/npy/ --mask_path ./checkpoint/generated_mask/
```

## Train Classifiers on the Pruned Dataset
```javascript
python train_subset.py --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 128 --save_path ./checkpoint/pruned-dataset --subset_rate 0.3  --target-probs-path ./generated/cifar10/42/target_probs_win_10_ep200.npy --score-path ./generated/cifar10/42/dual_mask_T30.npy --mask-path ./generated/cifar10/42/dual_mask_T30.npy --c_d 4 --sample beta --method dual
```

