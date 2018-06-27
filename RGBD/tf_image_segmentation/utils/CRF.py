import sys
from scipy.misc import imsave,imread
import numpy as np
from matplotlib import pyplot as plt
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

import skimage.io as io
fig_size = [15, 4]
plt.rcParams["figure.figsize"] = fig_size

	#final_probabilities = np.load('a.npy')
	#train_image = imread('train_image.jpg')
	#train_annotation = np.load('annotation.npy')
path = "/home/yaxing/softes/tensorflow_FCN/tensorflow_notes-master/dense_crf_python"
sys.path.append(path)

def crf(train_image,final_probabilities,train_annotation,number_class):

	for index_image in xrange(1):
		image = train_image
		softmax = final_probabilities[0].squeeze()
		#softmax_to_unary
		softmax = softmax.transpose((2, 0, 1))
		unary = unary_from_softmax(softmax)
		#softmax_to_unary
		unary = np.ascontiguousarray(unary)
		d = dcrf.DenseCRF(image.shape[0] * image.shape[1], number_class)
		d.setUnaryEnergy(unary)
		# This potential penalizes small pieces of segmentation that are
		# spatially isolated -- enforces more spatially consistent segmentations
		feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

		d.addPairwiseEnergy(feats, compat=3,
				    kernel=dcrf.DIAG_KERNEL,
				    normalization=dcrf.NORMALIZE_SYMMETRIC)
		# This creates the color-dependent features --
		# because the segmentation that we get from CNN are too coarse
		# and we can use local color features to refine them
		feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
						   img=image, chdim=2)

		d.addPairwiseEnergy(feats, compat=10,
				     kernel=dcrf.DIAG_KERNEL,
				     normalization=dcrf.NORMALIZE_SYMMETRIC)
		Q = d.inference(5)

		res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

		cmap = plt.get_cmap('bwr')

		f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
		ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
		ax1.set_title('Segmentation with CRF post-processing')
		probability_graph = ax2.imshow(np.dstack((train_annotation,)*3)*100)
		ax2.set_title('Ground-Truth Annotation')
		plt.savefig('annotation_%d.png'%index_image, bbox_inches='tight', pad_inches = 0)
		plt.gcf().clear()
		plt.show()
