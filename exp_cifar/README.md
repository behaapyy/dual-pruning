
---
## 🚀 Usage  
- **`exp_cifar`** for CIFAR (10, 100) experiments  

---
## Step 1: Collect Training Dynamics

This step is necessary to collect the training dynamics for subsequential coreset selection. DUAL only collects training dynamics during early 30 epochs.

```
python train.py \
    --data_path ./data \
    --dataset cifar100 \
    --arch resnet18 \
    --epochs 200 \
    --learning-rate 0.1 \
    --batch-size 128 \
    --manualSeed 42 \
    --dynamics \
    --save_path ./checkpoint
```

After this step, the training dynamics will be saved to the path specified by `--save_path`.

---

## Step 2: Evaluate Sample Importance

Once you have collected the training dynamics, you can compute an importance score for each data point. 

Run the following command, specifying the path to your saved dynamics and where you want to store the results.

```
python importance_evaluation.py \
    --dataset cifar100 \
    --dynamics_path ./checkpoint/cifar100/42/npy/ \
    --save_path ./checkpoint/cifar100/42/generated_mask/
```
This command generates two .npy files for each method:

- `XXX_score.npy`: Contains the importance score for each data point, ordered by original sample index.

- `XXX_mask.npy`: Contains the sorted sample indexes based on their importance scores.


---

## Step3: Train Classifiers on the Pruned Dataset

Now you can train a model using the pruned dataset that you created in the previous step. The `--subset_rate` parameter determines the percentage of data to keep. For example, a value of 0.3 keeps 30% of the dataset.

Use the following command, making sure to update the file paths (`--score-path`, `--mask-path`, and `--target-probs-path`) to the files generated in Step 2.

```
python train_subset.py \
    --data_path ./data \
    --dataset cifar100 \
    --arch resnet18 \
    --epochs 200 \
    --learning_rate 0.1 \
    --batch-size 128 \
    --save_path ./checkpoint/pruned-dataset/cifar100/42 \
    --subset_rate 0.3  \
    --target-probs-path ./checkpoint/cifar100/42/generated_mask/target_probs.npy \
    --score-path ./checkpoint/cifar100/42/generated_mask/dual_mask_T30.npy \
    --mask-path ./checkpoint/cifar100/42/generated_mask/dual_mask_T30.npy \
    --c_d 4 \
    --sample beta \
```


For an aggressive pruning rate, setting a smaller batch size will lead to better performance. We use batch size of 64 for 80% pruning, and 32 for 90% pruning. Please refer to the experimental settings section in our paper.




---
### Attribution

This code is mostly build upon 
```bibtex
@inproceedings{zhang2024TDDS,
  title={Spanning Training Progress: Temporal Dual-Depth Scoring (TDDS) for Enhanced Dataset Pruning},
  author={Xin, Zhang and Jiawei, Du and Yunsong, Li and Weiying, Xie and Joey Tianyi Zhou},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2024},
}
```
