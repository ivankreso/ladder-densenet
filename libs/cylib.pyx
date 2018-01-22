import numpy as np
cimport numpy as np
from libc cimport stdint

#np.import_array()

cdef extern from "cylib.h":
  ctypedef stdint.int64_t int64_t
  ctypedef stdint.uint64_t uint64_t
  ctypedef stdint.uint8_t uint8_t
  void add_confusion_matrix(int* y, int* yt, int size, uint64_t* matrix, int num_classes)
  void impl_convert_colors_to_ids(int num_classes, int* color_data, int width, int height,
                                  uint8_t* rgb_labels, uint8_t* id_labels, uint64_t* class_hist,
                                  float max_wgt, float* class_weights, float* weights)


def collect_confusion_matrix(y, yt, confusion_mat):
  cdef int size = y.size
  cdef int num_classes = confusion_mat.shape[0]
  cdef np.ndarray[int, mode="c", ndim=1] y_c = np.ascontiguousarray(y)
  cdef np.ndarray[int, mode="c", ndim=1] yt_c = np.ascontiguousarray(yt)
  cdef np.ndarray[uint64_t, mode="c", ndim=2] confusion_mat_c = confusion_mat
  add_confusion_matrix(&y_c[0], &yt_c[0], size, &confusion_mat_c[0,0], num_classes)


def convert_colors_to_ids(color_data, rgb_labels, id_labels, class_histogram, max_wgt,
                          class_weights, weights):
  cdef int num_classes = color_data.shape[0]
  cdef int width = rgb_labels.shape[1]
  cdef int height = rgb_labels.shape[0]
  cdef float max_wgt_c = max_wgt
  cdef np.ndarray[int, mode="c", ndim=2] color_data_c = color_data
  cdef np.ndarray[uint8_t, mode="c", ndim=3] rgb_labels_c = rgb_labels
  cdef np.ndarray[uint8_t, mode="c", ndim=2] id_labels_c = id_labels
  cdef np.ndarray[float, mode="c", ndim=2] weights_c = weights
  cdef np.ndarray[float, mode="c", ndim=1] class_weights_c = class_weights
  cdef np.ndarray[uint64_t, mode="c", ndim=1] class_hist_c = class_histogram
  impl_convert_colors_to_ids(num_classes, &color_data_c[0,0], width, height, &rgb_labels_c[0,0,0],
          &id_labels_c[0,0], &class_hist_c[0], max_wgt_c, &class_weights_c[0], &weights_c[0,0])

  #add_confusion_matrix(&y_c[0], &yt_c[0], size, &confusion_mat_c[0,0], num_classes)

#def collect_confusion_matrix(y, yt, conf_mat):
#    print(y.size)
#  for i in range(y.size):
#      l = y[i]
#    lt = yt[i]
#    if lt >= 0:
#        conf_mat[l,lt] += 1
#
