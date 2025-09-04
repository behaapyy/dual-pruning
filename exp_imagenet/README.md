
---
## 🚀 Usage  
**`exp_imagenet`** for ImageNet experiments  

---
## Train Classifiers on the Entire Dataset
This step is necessary to collect training dynamics for future coreset selection. DUAL only collects training dynamics during early 60 epochs.

```javascript
python train_imagenet.py --epochs 60 --lr 0.1 --scheduler cosine --task-name ICLR2023-ImageNet --base-dir /path/to/work-dir/imagenet/ --data-dir /dir/to/data/imagenet --network resnet34 --batch-size 256 --gpuid 0,1
```

## Sample Importance Evaluation
Using the training dynamics, you can get importance score of data points. 

Calculate DUAL, Dyn-Unc, TDDS score for each image
```javascript
python generate_importance_score_imagenet_dual.py --td-path /dir/to/td --save-path /path/to/save-dir
```
Calculate other baseline scores for each image
```javascript
python generate_importance_score_imagenet.py --data-dir /dir/to/data/imagenet --base-dir /path/to/work-dir/imagenet/ --task-name ImageNet-Score --data-score-path ./imagenet-data-score.pt
```

After the computation, you will obtain two .npy files (XXX_score.npy, XXX_mask.npy) storing scores ordered by sample indexes and sorted sample indexes by respective importance scores.

*For an aggressive pruning rate, setting a smaller batch size will lead to better performance. We use batch size of 64 for 80% pruning, and 32 for 90% pruning.

## Train Classifiers on the Pruned Dataset
#### DUAL
```javascript
python train_imagenet.py --iterations 300000 --iterations-per-testing 5000 --lr 0.1 --scheduler cosine --task-name dual --data-dir /dir/to/data/imagenet --base-dir /path/to/work-dir/imagenet/dual --coreset --coreset-mode dual --mask_npy_path save-path/mask_npy_path.npy --network resnet34 --batch-size 256 --coreset-ratio 0.1 --gpuid 0,1 --ignore-td
```

#### DUAL+Beta Sampling
```javascript
python train_imagenet.py --iterations 300000 --iterations-per-testing 5000 --lr 0.1 --scheduler cosine --task-name dual --data-dir /dir/to/data/imagenet --base-dir /path/to/work-dir/imagenet/dual --coreset --coreset-mode dual --mask_npy_path save-path/mask_npy_path.npy --score_npy_path save-path/score_npy_path.npy --probs_path save-path/target_probs.pt --network resnet34 --batch-size 256 --coreset-ratio 0.1 --gpuid 0,1 --ignore-td
```

This code is mostly build upon 
```bibtex
@misc{zheng2023coveragecentriccoresetselectionhigh,
      title={Coverage-centric Coreset Selection for High Pruning Rates}, 
      author={Haizhong Zheng and Rui Liu and Fan Lai and Atul Prakash},
      year={2023},
      eprint={2210.15809},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2210.15809}, 
}
```
