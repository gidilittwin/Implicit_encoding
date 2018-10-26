import h5py
import numpy as np
import os, sys, getopt
import pdb
class HDFDatasets:

    def __init__(self, outputpath, device_type = 'Unknown', phase = 'FW'):
        self.hdf_list = {}
        self.outputpath = outputpath
        self.phase = phase
        self.device_type = device_type
        
        

    def addDataset(self, name, dataset_size, num_of_elements_in_dataset, table_tuple,batch_size=5000):
        # tables_tuple: tuple of name and size of tables e.g. ('table_name',[1,96,96])
        print "Adding dataset {}".format(name)
        self.hdf_list[name] = {}
        hdf_c = self.hdf_list[name]
        hdf_c['Name'] = name
        hdf_c['HDF_SIZE'] = dataset_size
        hdf_c['tables'] = table_tuple
        hdf_c['TotalNumOfElements'] = num_of_elements_in_dataset
        hdf_c['batch_size'] = batch_size
    # def __init__(self):


    def createDataset(self):
        for key, hdf_prop in self.hdf_list.iteritems():
            hdf_prop['hdf_file_idx'] = 0
            hdf_prop['hdf_list_file_name'] = self.outputpath + '/' + hdf_prop['Name'] + '_hdf_list_' + self.device_type + "_" + self.phase + '.txt'
            hdf_prop['filename'] = self.outputpath + '/' + hdf_prop['Name'] + '_'
            hdf_prop['hdf_list_file_handle'] = open(hdf_prop['hdf_list_file_name'],'w')
            hdf_prop['hdf_entry_indx'] = 0
            hdf_prop['batch_entry_indx'] = 0
            hdf_prop['batches'] = {}

            self.createTables(hdf_prop)


    def createTables(self,c_hdf):
            if (c_hdf['hdf_entry_indx'] != 0):
                print "INcorrect hdf_entry_indx"
            c_hdf['table_handles'] = {}
            c_hdf['HDF_SIZE'] = min(c_hdf['HDF_SIZE'], c_hdf['TotalNumOfElements'])

            tmp_file_name = c_hdf['filename'] + self.device_type + "_" + self.phase + "_{0}.h5".format(c_hdf['hdf_file_idx'])
            c_hdf['hdf_file_idx']+=1
            c_hdf['file_handle'] = h5py.File(tmp_file_name,"w", driver='stdio')
            print "Creating new tables"
            for elmnt in c_hdf['tables']:
                table_sz = elmnt[1][:]
                chnk_sz = elmnt[1][:]
                table_sz.insert(0,c_hdf['HDF_SIZE'])
                chnk_sz.insert(0,1)
                chnk_sz = tuple(chnk_sz)
                if len(elmnt) == 3:
                    dtype = elmnt[2]
                else:
                    dtype = np.float32
                if len(elmnt) == 4:
                    gzip_opts = elmnt[3]
                else:
                    gzip_opts = 4
                c_hdf['table_handles'][elmnt[0]] = c_hdf['file_handle'].create_dataset(elmnt[0], table_sz, dtype=dtype, compression="gzip", compression_opts=gzip_opts, chunks=chnk_sz)

    def CheckStateOfHDF(self,remaining_elements):

        for key, hdf_prop in self.hdf_list.iteritems():
            # print "CheckingStateOfHDF of {} with {} remaining elements and {} index".format(hdf_prop['Name'],remaining_elements,hdf_prop['hdf_entry_indx'])
            if remaining_elements>0:
                print "About to closing file"
                hdf_prop['file_handle'].close()
                hdf_prop['file_handle'] = None
                prev_file_name = hdf_prop['filename'] + self.device_type + "_" + self.phase +"_{0}.h5".format(hdf_prop['hdf_file_idx']-1)
                print 'Closed '+ prev_file_name
                hdf_prop['hdf_list_file_handle'].write(os.path.abspath(prev_file_name) + '\n') #Add file to our hdf5 file list
                hdf_prop['hdf_entry_indx'] = 0
                self.createTables(hdf_prop)

    def PrepareBatches(self, batch_size):
        for key, hdf_prop in self.hdf_list.iteritems():
            hdf_prop['batches'] = {}
            hdf_prop['batch_size'] = batch_size
            for elmnt in hdf_prop['tables']:
                hdf_prop['batch_entry_indx'] = 0
                batch_size = elmnt[1][:]
                batch_size.insert(0,hdf_prop['batch_size']);
                # print "Creating Batch {} of size {}".format(elmnt[0],batch_size)
                hdf_prop['batches'][elmnt[0]] = (np.zeros(batch_size, dtype=float, order='C'))

    def SetBatch(self, dataset_name, data, first_axis_is_batch = False):
        if first_axis_is_batch:
            batch_size=0
            for table_name, table_data in data.iteritems():
                if batch_size==0:
                    # print 'Collecting batch size {} from {}'.format(table_data.shape[0],table_name)
                    batch_size = table_data.shape[0]
                else:
                    if batch_size != table_data.shape[0]:
                        raise NameError('Batch {} Sizes not equal {} vs. {}\n Total Size: {}'.format(table_name,batch_size,table_data.shape[0],table_data.shape ))
        else:
            batch_size = 1

        # print "Setting Batch for dataset {}".format(dataset_name)
        for table_name, table_data in data.iteritems():
            # if (table_data.shape[0]!= self.hdf_list[dataset_name]['batch_size']):
            #     print "Batch size does not match {} vs {}".format(table_data.shape[0],self.hdf_list[dataset_name]['batch_size'])
            # print "Setting batch {}".format(table_name)
            end_idx = self.hdf_list[dataset_name]['batch_entry_indx'] + batch_size
            self.hdf_list[dataset_name]['batches'][table_name][self.hdf_list[dataset_name]['batch_entry_indx']:end_idx,...] = table_data
        self.hdf_list[dataset_name]['batch_entry_indx'] +=batch_size


    def WriteBatch(self,dataset_name):
        # print "Writing Batch for dataset {}".format(dataset_name)

        c_dataset = self.hdf_list[dataset_name]
        remaining_elements = 0
        batch_start_indx = 0
        batch_end_indx = 0

        does_batch_fit = False # To enter while loop....
        remaining_elements = c_dataset['batch_size']
        while remaining_elements>0:
            hdf_start_indx = c_dataset['hdf_entry_indx']
            hdf_end_indx = min((hdf_start_indx+remaining_elements ),c_dataset['HDF_SIZE'])
            N_batch_elmnts_to_be_written = hdf_end_indx - hdf_start_indx
            batch_start_indx = batch_end_indx
            batch_end_indx = batch_start_indx + N_batch_elmnts_to_be_written
            if (N_batch_elmnts_to_be_written != remaining_elements): # Batch does not fit in HDF file
                print "Does not fit {} vs {}".format(N_batch_elmnts_to_be_written, remaining_elements)
                sys.stdout.write(';')
            else:
                sys.stdout.write('.')

            if N_batch_elmnts_to_be_written >0:
                for table_name, table_data in c_dataset['batches'].iteritems():
                    c_dataset['table_handles'][table_name][hdf_start_indx:hdf_end_indx,...] = table_data[batch_start_indx:batch_end_indx,...]
            sys.stdout.flush()
   
            remaining_elements -= N_batch_elmnts_to_be_written
            c_dataset['TotalNumOfElements'] -= N_batch_elmnts_to_be_written
            c_dataset['hdf_entry_indx'] += N_batch_elmnts_to_be_written

            self.CheckStateOfHDF(remaining_elements)




      
    def FinalizeDatasets(self):
        for key, hdf_prop in self.hdf_list.iteritems():
            if (hdf_prop['file_handle'] is not None):
                hdf_prop['file_handle'].close()
                prev_file_name = hdf_prop['filename'] + self.device_type + "_" + self.phase +"_{0}.h5".format(hdf_prop['hdf_file_idx']-1)
                print 'Closing '+ prev_file_name
                hdf_prop['hdf_list_file_handle'].write(os.path.abspath(prev_file_name) + '\n') #Add file to our hdf5 file list
                


        

