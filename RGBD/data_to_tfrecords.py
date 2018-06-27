from tf_image_segmentation.utils.tf_records import write_image_annotation_path_to_tfrecord
# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)

overall_train_image_annotation_filename_pairs = []

for file_index in xrange(1,16):
	for vedio_index in xrange(1000):
		if vedio_index < 10:
			vedio_index= '00%d'%vedio_index
		elif 9<vedio_index <100:
			vedio_index= '0%d'%vedio_index
		elif vedio_index>99:
			vedio_index= '%d'%vedio_index
		for image_index in xrange(10):
			labels_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/train/%d/labels/label_%d%s_%d.png'%(file_index,file_index,vedio_index,750*image_index)
			depths_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/train/%d/depths/depth_%d%s_%d.png'%(file_index,file_index,vedio_index,750*image_index)
			images_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/train/%d/photos/photo_%d%s_%d.jpg'%(file_index,file_index,vedio_index,750*image_index)
    			image_label_pairs = (images_path,labels_path,depths_path)
    			overall_train_image_annotation_filename_pairs.append(image_label_pairs)

for file_index in xrange(16,17):
	for vedio_index in xrange(864):
		if vedio_index < 10:
			vedio_index= '00%d'%vedio_index
		elif 9<vedio_index <100:
			vedio_index= '0%d'%vedio_index
		elif vedio_index>99:
			vedio_index= '%d'%vedio_index
		for image_index in xrange(10):
			labels_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/train/%d/labels/label_%d%s_%d.png'%(file_index,file_index,vedio_index,750*image_index)
			depths_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/train/%d/depths/depth_%d%s_%d.png'%(file_index,file_index,vedio_index,750*image_index)
			images_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/train/%d/photos/photo_%d%s_%d.jpg'%(file_index,file_index,vedio_index,750*image_index)
    			image_label_pairs = (images_path,labels_path,depths_path)
    			overall_train_image_annotation_filename_pairs.append(image_label_pairs)



write_image_annotation_path_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='./dataset/scene_net_augmented_train_16k.tfrecords')




