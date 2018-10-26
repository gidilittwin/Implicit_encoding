import h5py
import random
import numpy as np

class HDF5Iterator(object):
    """ Utility class for iterating over a list of HDF5 files, loading contents in batches, supporting shuffling
    """

    def __init__(self, h5listfile, keylist, batch_size, do_shuffling):
        """Initialize with the following parametes:
        h5listfile: name of file containing a list of HDF5 files - for each file, a single line containing the full path
        keylist: list of strings specifying the required fields. NOTE: assuming all fields have the same number of elements (shape(0))
        batch_size: size of batch returned on each call to 'get_batch'
        do_shuffling: whether to shuffle the file order and contents
        """
        self.batch_size = batch_size
        self.do_shuffling = do_shuffling

        # initialize the file list
        if isinstance(h5listfile, basestring):
            if h5listfile.endswith('.h5'):
                self.h5list = [h5listfile]
            else:
                self.h5list = [line.rstrip('\n') for line in open(h5listfile)]
        elif isinstance(h5listfile, list) and isinstance(h5listfile[0], basestring):
            self.h5list = h5listfile
        self.file_perm = range(0, len(self.h5list)) # list of file indices in permuted order
        if self.do_shuffling:
            random.shuffle(self.file_perm)
        self.curr_file_ind = -1 # current index to permuted file list
        self.H = None # h5 object handle
        self.element_perm = None # list of permuted order of elements in current file
        self.curr_elem_ind = 0 # current index to permuted element indices
        self.cache_size = 512
        self.cache_counter = self.cache_size
        self.keylist = self.__check_keys(keylist)
        self.__prepare_next_file()
        self.epoch = 0 # number of epoches read
        self.read_counter = 0 # number of elements read

    def reset(self, reshuffle):
        """reset the iterator to the first element, optionally reshuffling if specified"""
        if self.do_shuffling and reshuffle:
            random.shuffle(self.file_perm)
        self.curr_file_ind = -1 # current index to permuted file list
        self.element_perm = None # list of permuted order of elements in current file
        self.curr_elem_ind = 0 # current index to permuted element indices
        self.read_counter = 0 # number of elements read
        self.__prepare_next_file()      

    def num_elements_in_epoch(self):
        """counts the number of elements in all the input files.
        NOTE: this is an expensive operation, because the number is not cached"""
        k = self.keylist[0]
        num_elements = 0
        for f in self.h5list:
            H = h5py.File(f)
            num_elements += H[k].shape[0]
        return num_elements

    def num_epoches_read(self):
        """returns the number of times the entire file list has been traversed"""
        return self.epoch

    def num_elements_read(self):
        """returns the number of elements read in the current epoch"""
        return self.read_counter

    def get_batch(self, batch_size=-1):
        """get the next batch of specified elements, wrapping around when reaching the end of the last file (optionally reshuffling).
        if specified, the given batch size is used, otherwise the default size is used
        returns: a list of numpy arrays in the order of the keylist
        """
        if batch_size == -1:
            batch_size = self.batch_size
        num_keys = len(self.keylist)
        values = num_keys*[None]
        for i in xrange(num_keys):
            k = self.keylist[i]
            vshape = self.H[k].shape
            values[i] = np.empty([batch_size]+list(vshape[1:]), dtype=self.H[k].dtype)
            
        num_loaded = 0
#        elem_ids = np.zeros((self.batch_size,),dtype=np.int64)
        elem_ids = []
        while num_loaded < batch_size:
            num_remaining = batch_size - num_loaded
            actual_batch_size = np.min([len(self.element_perm) - self.curr_elem_ind, num_remaining])
            elem_indices = np.sort(self.element_perm[self.curr_elem_ind:self.curr_elem_ind+actual_batch_size])
            for i in xrange(num_keys):
                k = self.keylist[i]
                v = values[i]
                if actual_batch_size == 1: # avoid the problem made by unwanted squeezing of first dimenstion when it is 1
                    v[num_loaded:num_loaded+actual_batch_size,:] = self.H[k][[self.element_perm[self.curr_elem_ind]],:]
                else:
                    v[num_loaded:num_loaded+actual_batch_size,:] = self.H[k][elem_indices,:]
            elem_ids.append(elem_indices)
            num_loaded += actual_batch_size
            self.curr_elem_ind += actual_batch_size
            self.read_counter += actual_batch_size
            self.cache_counter -= actual_batch_size
            if self.curr_elem_ind == len(self.element_perm):
                self.__prepare_next_file()  
            # self.__prepare_cache()
        elem_ids = np.concatenate(elem_ids,0) 
        values.append(np.expand_dims(elem_ids,-1))
        return values

    def get_batch_by_idx(self,
                         start_idx,
                         h5file_index=0,
                         batch_size=-1):

        if batch_size == -1:
            batch_size = self.batch_size
        if self.curr_file_ind != h5file_index:
            if 0 == h5file_index:
                self.curr_file_ind = len(self.h5list)-1
            else:
                self.curr_file_ind = h5file_index-1

            self.__prepare_next_file()

        k = self.keylist[0]
        curr_h5_len = self.H[k].shape[0]

        actual_batch_size = min(batch_size, (curr_h5_len-start_idx))
        num_keys = len(self.keylist)
        values = num_keys*[None]
        for i in xrange(num_keys):
            k = self.keylist[i]
            vshape = self.H[k].shape
            values[i] = np.empty([actual_batch_size]+list(vshape[1:]), dtype=self.H[k].dtype)

        for i in xrange(num_keys):
            k = self.keylist[i]
            v = values[i]

            v[:actual_batch_size, :] = self.H[k][start_idx:(start_idx+actual_batch_size), :]

        return values

    @property
    def actual_keylist(self):
        return self.keylist
            
    def __prepare_cache(self):
        if self.cache_counter <= 0:
            first_elem_ind = self.curr_elem_ind
            last_elem_ind = np.minimum(self.curr_elem_ind + self.cache_size, len(self.element_perm))
            print 'reading cache for elements {}-{}'.format(first_elem_ind, last_elem_ind)
            actual_cache_size = last_elem_ind - first_elem_ind
            elem_indices = np.sort(self.element_perm[first_elem_ind:last_elem_ind])
            if actual_cache_size > 1:
                for k in self.keylist:
                    print 'reading cache for key {}'.format(k)
                    cache = self.H[k][elem_indices,:]
                self.cache_counter = actual_cache_size

    def __check_keys(self, keylist):
        actaul_keys = []
        if len(self.h5list) > 0:
            with h5py.File(self.h5list[0], 'r') as H:
                for k in keylist:
                    if k in H.keys():
                        actaul_keys += [k]

        return actaul_keys

    def __prepare_next_file(self):
        self.curr_file_ind += 1
        if self.curr_file_ind == len(self.file_perm):
            self.epoch += 1
            self.read_counter = 0
            if self.do_shuffling:
                random.shuffle(self.file_perm)
            self.curr_file_ind = 0

        if (self.H):
            self.H.close()
        file_ind = self.file_perm[self.curr_file_ind]
        print 'opening file {}'.format(self.h5list[file_ind])
        self.H = h5py.File(self.h5list[file_ind], 'r')

        # prepare shuffled list of elements
        key = self.keylist[0]
        num_elements = self.H[key].shape[0]
        for k in self.keylist:
            assert (self.H[k].shape[0] == num_elements)  # all element types must have the same number of elements
        self.element_perm = range(num_elements)
        if self.do_shuffling:
            random.shuffle(self.element_perm)
        self.curr_elem_ind = 0

        self.cache_counter = 0
        # self.__prepare_cache()