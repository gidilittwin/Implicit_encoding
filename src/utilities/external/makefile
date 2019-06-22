nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

all: tf_nndistance_so.so tf_nndistance_g.cu.o


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(TF_INC) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
