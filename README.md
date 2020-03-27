# multi-label-classifications
几种多标签分类实现方式


语言：python2.7  


框架：pytorch


MIL/  为multiple instance learning参考[https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Fang_From_Captions_to_2015_CVPR_paper.pdf]


multi_label_with_focal_loss/  
为基于sigmoid的多标签分类，并添加focal loss函数进行训练



其中两个目录都包含的了mAP评测方法，实现了11-point interpolated average precision 和 更加准确的一种方法
