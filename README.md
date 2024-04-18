# Two Tricks to Improve Unsupervised Segmentation Learning
Source code for [Two Tricks to Improve Unsupervised Segmentation Learning
](https://arxiv.org/abs/2404.03392).
## Dependencies
This implementation depends on pytorch. Visit [pytorch website](https://pytorch.org/get-started/locally/) to 
get the latest version. Use the following command for additional requirements

```
pip install -r requirements.txt
```

## Training
To train the baseline on duts run the command.
```
python sempart_main.py --batch_size=8 --lr=1e-4 --img_size=320 --weight_entropy=1 
--weight_reg_img=1 --weight_reg_feat=1 --weight_const=0.01 --weight_mask_s=0.01 
--weight_mask=1 --n_last_blocks=1 --save_mask 
--experiment_name=sempart_exp1 --dataset=duts --save_frequency=10 --epochs=30 
--patch_size=8 --use_keys --ncut_thr=0.4
```
To train the the improved version on duts run the command.
```
python sempart_main.py --batch_size=8 --lr=1e-4 --img_size=320 --weight_entropy=1 
--weight_reg_img=1 --weight_reg_feat=1 --weight_const=0.01 --weight_mask_s=0.01 
--weight_mask=1 --n_last_blocks=1 --save_mask 
--experiment_name=sempart_exp1 --dataset=duts --save_frequency=10 --epochs=30 
--patch_size=8 --use_keys --ncut_thr=0.4 --use_multi_scale --use_gf
```

## Evaluation
To evaluate a pretrained segmentation head run the command. Set *--ckpt_segmenter_path* to 
the desired pretrained segmentation head. Add *--use_gf* to utilise guided filtering during the evaluation.
Change *--dataset* to *duts*, *dut-omron*, or *ecssd*
```
python sempart_main.py --batch_size=8 --img_size=320 --dataset=duts 
--patch_size=8 --use_keys --ckpt_segmenter_path=path/to/segmentation/head
```

## Pretained Models
You can download the Sempart model trained with 

| MODEL                              |                                                                                            |
|------------------------------------|--------------------------------------------------------------------------------------------|
| Sempart                            | [Link](https://drive.google.com/file/d/1d8HpA-8kCKC1V7aJxh4ayFXzzSOcSjFU/view?usp=sharing)                                                                                   |
| Sempart w/ Multi-Scale consistency | [Link](https://drive.google.com/file/d/1GsFCI9mF3p7HdyFZsgPLBIJcZsCRSV-e/view?usp=sharing) |

