	ƅ!??@ƅ!??@!ƅ!??@	,???:k?,???:k?!,???:k?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ƅ!??@??G6W???A?;???@Y??F?????*	-??????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?=??j:5@!U???N?X@)?=??j:5@1U???N?X@:Preprocessing2F
Iterator::Modelٓ??<??!?a?i????)???w????1??q0?S??:Preprocessing2P
Iterator::Model::Prefetch?5?U?ő?!??:?<???)?5?U?ő?1??:?<???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapP?,?;5@!??<??X@)??"?-?r?1?????7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9,???:k?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??G6W?????G6W???!??G6W???      ??!       "      ??!       *      ??!       2	?;???@?;???@!?;???@:      ??!       B      ??!       J	??F???????F?????!??F?????R      ??!       Z	??F???????F?????!??F?????JCPU_ONLYY,???:k?b 