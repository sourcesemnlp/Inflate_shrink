# Evaluation on MSCOCO-1k
## Evaluate an Inflate model

```
CUDA_VISIBLE_DEVICES=0 python eval.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE inflate \
--margin 0.6 \
--data_path path_to_datasets  \
--resume path_to_trained_inflated_model (downloaded inflate.pth.tar)
```
## Evaluate an Inflate->Base model

```
CUDA_VISIBLE_DEVICES=0 python eval.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE shrinktobase \
--margin 0.6 \
--data_path path_to_datasets  \
--resume path_to_trained_inflated_model   (downloaded inflate_base.pth.tar)
```

## Evaluate an Inflate->Base->Fast model

```
CUDA_VISIBLE_DEVICES=0 python eval.py --task retrieval \
--data_name coco_precomp --batch_size 128 \
--learning_rate 0.0001 \
--lr_update 20 \
--num_epochs 40 \
--log_step 200 \
--workers 16 \
--MODE shrinktofast \
--margin 0.6 \
--data_path path_to_datasets  \
--resume path_to_distilled_base_model   (downloaded inflate_fast.pth.tar)
```

