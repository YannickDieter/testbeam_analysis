# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libcpp cimport bool as cpp_bool  # to be able to use bool variables, as cpp_bool according to http://code.google.com/p/cefpython/source/browse/cefpython/cefpython.pyx?spec=svne037c69837fa39ae220806c2faa1bbb6ae4500b9&r=e037c69837fa39ae220806c2faa1bbb6ae4500b9
from data_struct cimport numpy_hit_info, numpy_cluster_hit_info, numpy_cluster_info

cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error

cdef extern from "Basis.h":
    cdef cppclass Basis:
        Basis()

cdef extern from "Clusterizer.h":
    cdef cppclass HitInfo:
        HitInfo()
    cdef cppclass ClusterHitInfo:
        ClusterHitInfo()
    cdef cppclass ClusterInfo:
        ClusterInfo()
    cdef cppclass Clusterizer(Basis):
        Clusterizer(unsigned int maxCol, unsigned int maxRow) except +
        void setErrorOutput(cpp_bool pToggle)
        void setWarningOutput(cpp_bool pToggle)
        void setInfoOutput(cpp_bool pToggle)
        void setDebugOutput(cpp_bool pToggle)

        void addHits(HitInfo *& rHitInfo, const unsigned int & rNhits) except +

        void createClusterHitInfoArray(cpp_bool toggle)
        void createClusterInfoArray(cpp_bool toggle)

        void setClusterHitInfoArray(ClusterHitInfo *& rClusterHitInfo, const unsigned int & rSize)
        void setClusterInfoArray(ClusterInfo *& rClusterHitInfo, const unsigned int & rSize)
        void setXclusterDistance(const unsigned int & pDx)
        void setYclusterDistance(const unsigned int & pDy)
        void setFrameclusterDistance(const unsigned int & pdFrame)
        void setMinClusterHits(const unsigned int & pMinNclusterHits)
        void setMaxClusterHits(const unsigned int & pMaxNclusterHits)
        void setMaxClusterHitCharge(const unsigned int & pMaxClusterHitCharge)

        void setMaxHitCharge(const unsigned int & pMaxHitCharge)

        void getClusterSizeHist(unsigned int & rNparameterValues, unsigned int *& rClusterSize, cpp_bool copy)
        void getClusterChargeHist(unsigned int & rNparameterValues, unsigned int *& rClusterCharge, cpp_bool copy)

        # void clusterize()

        unsigned int getNclusters()

        void reset()
        void test()

cdef class HitClusterizer:
    cdef Clusterizer * thisptr  # hold a C++ instance which we're wrapping
    def __cinit__(self, n_columns=1000, n_rows=1000):
        self.thisptr = new Clusterizer(<unsigned int> n_columns, <unsigned int> n_rows)
    def __dealloc__(self):
        del self.thisptr
    def set_debug_output(self, toggle):
        self.thisptr.setDebugOutput(< cpp_bool > toggle)
    def set_info_output(self, toggle):
        self.thisptr.setInfoOutput(< cpp_bool > toggle)
    def set_warning_output(self, toggle):
        self.thisptr.setWarningOutput(< cpp_bool > toggle)
    def set_error_output(self, toggle):
        self.thisptr.setErrorOutput(< cpp_bool > toggle)
    def add_hits(self, cnp.ndarray[numpy_hit_info, ndim=1] hit_info):
        self.thisptr.addHits(< HitInfo *&> hit_info.data, < unsigned int > hit_info.shape[0])
    def create_cluster_hit_info_array(self, value=True):
        self.thisptr.createClusterHitInfoArray(< cpp_bool > value)
    def create_cluster_info_array(self, value=True):
        self.thisptr.createClusterInfoArray(< cpp_bool > value)
    def set_cluster_hit_info_array(self, cnp.ndarray[numpy_cluster_hit_info, ndim=1] cluster_hit_info):
        self.thisptr.setClusterHitInfoArray(< ClusterHitInfo *&> cluster_hit_info.data, < const unsigned int &> cluster_hit_info.shape[0])
    def set_cluster_info_array(self, cnp.ndarray[numpy_cluster_info, ndim=1] cluster_info):
        self.thisptr.setClusterInfoArray(< ClusterInfo *&> cluster_info.data, < const unsigned int &> cluster_info.shape[0])
    def set_x_cluster_distance(self, value):
        self.thisptr.setXclusterDistance(< const unsigned int &> value)
    def set_y_cluster_distance(self, value):
        self.thisptr.setYclusterDistance(< const unsigned int &> value)
    def set_frame_cluster_distance(self, value):
        self.thisptr.setFrameclusterDistance(< const unsigned int &> value)
    def set_min_cluster_hits(self, value):
        self.thisptr.setMinClusterHits(< const unsigned int &> value)
    def set_max_cluster_hits(self, value):
        self.thisptr.setMaxClusterHits(< const unsigned int &> value)
    def set_max_cluster_hit_charge(self, value):
        self.thisptr.setMaxClusterHitCharge(< const unsigned int &> value)
    def set_max_charge(self, value):
        self.thisptr.setMaxHitCharge(<const unsigned int &> value)
    def get_cluster_size_hist(self, cnp.ndarray[cnp.uint32_t, ndim=1] cluster_size_hist, value=True):
        rNparameterValues = 0
        self.thisptr.getClusterSizeHist(< unsigned int &> rNparameterValues, < unsigned int *&> cluster_size_hist.data, < cpp_bool > value)
        return rNparameterValues
    def get_cluster_charge_hist(self, cnp.ndarray[cnp.uint32_t, ndim=1] cluster_charge_hist, value=True):
        rNparameterValues = 0
        self.thisptr.getClusterChargeHist(< unsigned int &> rNparameterValues, < unsigned int *&> cluster_charge_hist.data, < cpp_bool > value)
        return rNparameterValues
    def get_n_clusters(self):
        return < unsigned int > self.thisptr.getNclusters()
    def reset(self):
        self.thisptr.reset()
    def test(self):
        self.thisptr.test()