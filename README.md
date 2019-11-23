## FOTS_TF(端到端的文本识别-nba记分牌识别)
### 1. custom训练数据
最终数据需要的形式是每个图片对应一个txt包含每一个bbox的（xyxyxyxy,gt）这样的label数据，
因此第一步首先把标注数据的csv转成一个一个的txt。
eg：即把nba_train_1023.csv转为training_gt_1080p_v1106
``` python
python get_custom_sbb.py
``` 
### 2. train
``` python
#!/bin/sh
python /FOTS_TF/main_train.py \
--batch_size_per_gpu=16 \
--num_readers=6 \
--gpu_list='0' \
--restore=False \
--checkpoint_path='checkpoints/bs16_1080p_v1106_aughsv/' \
--pretrained_model_path='models/model.ckpt-733268' \
--training_data_dir='training_img_1080p_v1106' \
--training_gt_data_dir='training_gt_1080p_v1106'
``` 
其中，checkpoint_path为要保存的模型的路径；pretrained_model_path为加载icdar的预训练模型路径。

### 3. infer
``` python
#!/bin/sh
python main_test_bktree.py \
--test_data_path='samples' \
--checkpoint_path='checkpoints/bs16_540p_v1106_aughsv/' \
--output_dir='outputs/outputs_bs16_540p_v1106_aughsv_2016' 
``` 

### 4. eval
在这一部分，我们用了后处理逻辑，然后用来评估test集的准确率，大家不需要这一部分，可以忽略。
``` python
#!/bin/sh
python /data/ceph_11015/ssd/anhan/nba/FOTS_TF/main_test_bktree_eval_v2.py \
--just_infer=False \
--check_teamname=False \
--test_data_path='/data/ceph_11015/ssd/templezhang/scoreboard/EAST/data/check_res_15161718_test_null.csv' \
--checkpoint_path='/data/ceph_11015/ssd/anhan/nba/FOTS_TF/checkpoints/bs16_540p_v1106_aughsv/' \
--output_dir='/data/ceph_11015/ssd/anhan/nba/FOTS_TF/outputs/outputs_bs16_540p_v1106_aughsv_eval' 
``` 