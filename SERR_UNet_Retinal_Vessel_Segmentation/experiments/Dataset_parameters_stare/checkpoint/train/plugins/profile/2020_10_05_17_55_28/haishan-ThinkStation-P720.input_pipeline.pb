	+N???@+N???@!+N???@	?s?"?Ao??s?"?Ao?!?s?"?Ao?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+N???@??DR???A,????@Y??K?[??*	?l????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??7??#@!??f?X@)??7??#@1??f?X@:Preprocessing2F
Iterator::Models???M??!G#gSV???)??j?T??1U.H,#??:Preprocessing2P
Iterator::Model::Prefetch??o???!p??^????)??o???1p??^????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?o?h??#@!ݘ???X@)?PۆQp?1$???Tؤ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?s?"?Ao?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??DR?????DR???!??DR???      ??!       "      ??!       *      ??!       2	,????@,????@!,????@:      ??!       B      ??!       J	??K?[????K?[??!??K?[??R      ??!       Z	??K?[????K?[??!??K?[??JCPU_ONLYY?s?"?Ao?b 