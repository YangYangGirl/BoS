DATASET=self_driving2coco
ANN_DIR=./data/kaggle/self_driving2coco/self_driving2coco_sample_img_250.json

POS_ARRAY="1 2"
POS_NAME="1_2"
DROP=0.15

TAG1=cost_droprate_0_15_${DATASET}_droppos_${POS_NAME}_s250_n50
for i in `seq 0 49`;
  do
  if [ -f "./res_iou_ac_area_filter/${TAG1}/${i}.json" ]; then
    echo "pass"
  else
    echo "./res_iou_ac_area_filter/${TAG1}/${i}.json not exist"
    python tools/test.py configs/retinanet/retinanet_r50_fpn_1x_coco_car_uni_size.py ./work_dir/retinanet/retinanet_r50_fpn_1x_coco_car/epoch_36.pth --eval bbox --cfg-option data.test.img_prefix=./data/kaggle/${DATASET}/meta/${i} data.test.ann_file=${ANN_DIR}  --tag1 ${TAG1} --tag2 ${i} --dropout_uncertainty ${DROP} --drop_layers ${POS_ARRAY}
  fi
done
ls -l ./res_iou_ac_area_filter/${TAG1}/* | grep "^-" | wc -l
