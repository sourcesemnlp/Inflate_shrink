# Training
## Training on COCO
### Train a Base Model

```
CUDA_VISIBLE_DEVICES=0 python train.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE base \
--margin 0.6 \
--data_path path_to_datasets 
```
### Train a Fast Model

```
CUDA_VISIBLE_DEVICES=0 python train.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE fast \
--margin 0.2 \
--data_path path_to_datasets 
```
### Train a Inflate Model

```
CUDA_VISIBLE_DEVICES=0 python train.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE inflate \
--margin 0.6 \
--data_path path_to_datasets 
```

### Shrink Inflate -> Base

```
CUDA_VISIBLE_DEVICES=0 python train.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE shrinktobase \
--margin 0.6 \
--data_path path_to_datasets \
--resume path_to_trained_inflated_model
```

### Shrink Inflate -> Base -> Fast

```
CUDA_VISIBLE_DEVICES=0 python train.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE shrinktobase \
--margin 0.6 \
--data_path path_to_datasets \
--resume path_to_distilled_base_model
```