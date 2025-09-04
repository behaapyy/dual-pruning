
---
## 🚀 Usage  
- **`exp_cifar`** for CIFAR (10, 100) experiments  

---
## Train Classifiers on the Entire Dataset
This step is necessary to collect the training dynamics for subsequential coreset selection. DUAL only collects training dynamics during early 30 epochs.

```javascript
python train.py --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 128 --dynamics --save_path ./checkpoint/all-dataset
```

## Sample Importance Evaluation
Using the training dynamics, you can get importance score of data points. 

```javascript
python importance_evaluation.py --dynamics_path ./checkpoint/all-dataset/npy/ --mask_path ./checkpoint/generated_mask/
```
After the computation, you will obtain two .npy files (XXX_score.npy, XXX_mask.npy) storing scores ordered by sample indexes and sorted sample indexes by respective importance scores.

*For an aggressive pruning rate, setting a smaller batch size will lead to better performance. We use batch size of 64 for 80% pruning, and 32 for 90% pruning.

## Train Classifiers on the Pruned Dataset
```javascript
python train_subset.py --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 128 --save_path ./checkpoint/pruned-dataset --subset_rate 0.3  --target-probs-path ./generated/cifar10/42/target_probs_win_10_ep200.npy --score-path ./generated/cifar10/42/dual_mask_T30.npy --mask-path ./generated/cifar10/42/dual_mask_T30.npy --c_d 4 --sample beta --method dual
```

---
This code is mostly build upon 
```bibtex
@inproceedings{zhang2024TDDS,
  title={Spanning Training Progress: Temporal Dual-Depth Scoring (TDDS) for Enhanced Dataset Pruning},
  author={Xin, Zhang and Jiawei, Du and Yunsong, Li and Weiying, Xie and Joey Tianyi Zhou},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2024},
}
```
