nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64/
tensorflow = /home/gidi/.local/lib/python2.7/site-packages/tensorflow/include
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

all: tf_nndistance_so.so render_balls_so.so
.PHONY : all

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


