
"""
Copyright (c) 2020. All rights reserved.
Created by lixiang on 2020/5/11
"""

from template.infers.segmention_infer import SegmentionInfer
from template.metric.metric import *
from conf.utils.config_utils import process_config


repredict=True

def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = process_config('conf/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    if repredict==True:

        print('[INFO] Predicting...')
        infer = SegmentionInfer( config)
        infer.predict()

    print('[INFO] Metric results...')
    gtlist=fileList(config.test_gt_path,'*'+config.test_gt_datatype)
    problist=fileList(config.test_result_path,'*.bmp')
    modelName=['SERR-U-Net']
    drawCurve(gtlist,[problist],modelName,'STARE',config.checkpoint)

    print('[INFO] Fininshed...')


if __name__ == '__main__':
    main_test()
