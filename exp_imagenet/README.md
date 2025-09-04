
---
## 🚀 Usage  
**`exp_imagenet`** for ImageNet experiments  

---
## Train Classifiers on the Entire Dataset
This step is necessary to collect training dynamics for future coreset selection. DUAL only collects training dynamics during early 60 epochs.

```javascript
python train_imagenet.py --epochs 60 --lr 0.1 --scheduler cosine --task-name imagenet --base-dir /path/to/work-dir/imagenet/ --data-dir /dir/to/data/imagenet --network resnet34 --batch-size 256 --gpuid 0,1
```

## Sample Importance Evaluation
Using the training dynamics, you can get importance score of data points. 

Calculate DUAL, Dyn-Unc, TDDS score for each image
```javascript
python generate_importance_score_imagenet_dual.py --td-path /dir/to/td --save-path /path/to/save-dir
```
After the computation, you will obtain two .npy files (XXX_score.npy, XXX_mask.npy) storing scores ordered by sample indexes and sorted sample indexes by respective importance scores.

Calculate other baseline scores for each image
```javascript
python generate_importance_score_imagenet.py --data-dir /dir/to/data/imagenet --base-dir /path/to/work-dir/imagenet/ --task-name imagenet --data-score-path ./imagenet-data-score.pt
```

*For an imagenet experience, we don't set a smaller batch size for aggressive pruning. 

## Train Classifiers on the Pruned Dataset
#### DUAL
```javascript
python train_imagenet.py --iterations 300000 --iterations-per-testing 5000 --lr 0.1 --scheduler cosine --task-name dual --data-dir /dir/to/data/imagenet --base-dir /path/to/work-dir/imagenet/dual --coreset --coreset-mode dual --mask_npy_path save-path/mask_npy_path.npy --network resnet34 --batch-size 256 --coreset-ratio 0.1 --gpuid 0,1 --ignore-td
```

#### DUAL+Beta Sampling
```javascript
python train_imagenet.py --iterations 300000 --iterations-per-testing 5000 --lr 0.1 --scheduler cosine --task-name dual --data-dir /dir/to/data/imagenet --base-dir /path/to/work-dir/imagenet/dual --coreset --coreset-mode dual --mask_npy_path save-path/mask_npy_path.npy --score_npy_path save-path/score_npy_path.npy --probs_path save-path/target_probs.pt --network resnet34 --batch-size 256 --coreset-ratio 0.1 --gpuid 0,1 --ignore-td
```


---
This code is mostly build upon 
```bibtex
@article{zheng2022coverage,
  title={Coverage-centric coreset selection for high pruning rates},
  author={Zheng, Haizhong and Liu, Rui and Lai, Fan and Prakash, Atul},
  journal={arXiv preprint arXiv:2210.15809},
  year={2022}
}
```
