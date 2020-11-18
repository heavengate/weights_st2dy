# weights_st2dy


1. `layers.py` 替换paddle包中 `paddle/fluid/dygraph/layers.py`, `layer_helper_base` 替换paddle包中 `paddle/fluid/layer_helper_base.py`, 可以通过`pip install/uninstall`获取paddle包安装路径，替换过程建议用vimdiff把增量代码替换过去。

2. 准备PaddleDetection静态图和动态图的代码库, `static_print.py` 移动至静态图库中，`dynamic_print.py` 移动至动态图库中, 运行对应配置文件并将输出保存为文件，如：


```
# 静态图库中
python static_print.py -c configs/yolov3_darknet.yml 2>&1 | tee yolov3_st_print.txt
```

```
# 动态图库中
python static_print.py -c configs/yolov3_darknet53_270e_coco.yml 2>&1 | tee yolov3_dy_print.txt
```

3. 运行`parse.py`如下，会解析静态图和动态图权重名对应关系，并将结果保存为`weight_name_map.txt`

```
python parse.py yolov3_dy_print.txt yolov3_st_print.txt
```

若出现无法匹配的权重名，会有如下提示，手动选择匹配的权重名继续匹配。

```
bn2c_branch2c_scale                                matched      backbone.res2.res2c.branch2c.norm.weight
bn2c_branch2c_offset                               matched      backbone.res2.res2c.branch2c.norm.bias
bn2c_branch2c_mean                                 matched      backbone.res2.res2c.branch2c.norm._mean
bn2c_branch2c_variance                             matched      backbone.res2.res2c.branch2c.norm._variance
res3a_branch2a_weights                             matched      backbone.res3.res3a.branch2a.conv.weight
bn3a_branch2a_scale                                matched      backbone.res3.res3a.branch2a.norm.weight
bn3a_branch2a_offset                               matched      backbone.res3.res3a.branch2a.norm.bias
bn3a_branch2a_mean                                 matched      backbone.res3.res3a.branch2a.norm._mean
bn3a_branch2a_variance                             matched      backbone.res3.res3a.branch2a.norm._variance
('*****match wrong*******', ('res3a_branch2b_weights', [128, 128, 3, 3]), ('weight', [512, 256, 1, 1], 'backbone.res3.res3a.short.conv.weight'), ', is ConvBN/FC block: ', 1)
Please select dygraph weight name for res3a_branch2b_weights:
         1. backbone.res3.res3a.branch2b.conv.weight
         2. backbone.res3.res3b.branch2b.conv.weight
         3. backbone.res3.res3c.branch2b.conv.weight
         4. backbone.res3.res3d.branch2b.conv.weight
selection:
```

如上输出，则为 `res3a_branch2b_weights` 判断后手动选择1，即`backbone.res3.res3a.branch2b.conv.weight`继续匹配

**注意：** 解析完成后最好自行check一下`weight_name_map.txt`中的权重对应关系，如`mask_rcnn_fpn`中优于fpn计算顺序动态图和静态图相反，会存在如下情况，可手动修改下`weight_name_map.txt`

```
fpn_inner_res2_sum_lateral_w                       neck.fpn_inner_res2_sum_lateral.weight
fpn_inner_res2_sum_lateral_b                       neck.fpn_inner_res2_sum_lateral.bias
#  -------------------- 2, 3, 4, 5 顺序反了 --------------------
fpn_res5_sum_w                                     neck.fpn_res2_sum.weight
fpn_res5_sum_b                                     neck.fpn_res2_sum.bias
fpn_res4_sum_w                                     neck.fpn_res3_sum.weight
fpn_res4_sum_b                                     neck.fpn_res3_sum.bias
fpn_res3_sum_w                                     neck.fpn_res4_sum.weight
fpn_res3_sum_b                                     neck.fpn_res4_sum.bias
fpn_res2_sum_w                                     neck.fpn_res5_sum.weight
fpn_res2_sum_b                                     neck.fpn_res5_sum.bias
#  ---------------------------------------------------------
conv_rpn_fpn2_w                                    rpn_head.rpn_feat.rpn_conv.weight
conv_rpn_fpn2_b                                    rpn_head.rpn_feat.rpn_conv.bias
rpn_cls_logits_fpn2_w                              rpn_head.rpn_rois_score.weight
rpn_cls_logits_fpn2_b                              rpn_head.rpn_rois_score.bias
```

4. 使用`convert.py` 和 3 中生成的 `weight_name_map.txt` 完成权重转换，命令如下：

```
export PYTHONPATH=$PYTHONPATH:<path/to/PaddleDetection>
python convert.py https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar weight_name_map.txt new.pdparams
```

`new.pdparams` 即为转换后的动态图权重
