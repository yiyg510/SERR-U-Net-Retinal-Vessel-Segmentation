	ZKi???@ZKi???@!ZKi???@	??m?(?q???m?(?q?!??m?(?q?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ZKi???@2 Ǟ=??A?$"?{??@Y???;?_??*	??ο??@2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorcFx{B4@!????X@)cFx{B4@1????X@:Preprocessing2F
Iterator::ModelF??}ȣ?!R[? #\??)B??K8???1?[7??C??:Preprocessing2P
Iterator::Model::Prefetch???Ü??!?Z	`?t??)???Ü??1?Z	`?t??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?oD??C4@!ӯo???X@)獓¼?y?1՝:????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??m?(?q?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2 Ǟ=??2 Ǟ=??!2 Ǟ=??      ??!       "      ??!       *      ??!       2	?$"?{??@?$"?{??@!?$"?{??@:      ??!       B      ??!       J	???;?_?????;?_??!???;?_??R      ??!       Z	???;?_?????;?_??!???;?_??JCPU_ONLYY??m?(?q?b 