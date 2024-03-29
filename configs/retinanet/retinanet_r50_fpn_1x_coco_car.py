_base_ = [
    '../_base_/models/retinanet_r50_fpn_car.py',
    '../_base_/datasets/coco_detection_car.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
optimizer = dict(_delete_=True, type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)