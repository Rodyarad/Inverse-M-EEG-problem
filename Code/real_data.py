import numpy as np
import mne
from mne.datasets import sample

data_path = sample.data_path()
subject = 'sample'
evoked_fname = data_path / 'MEG' / subject / 'sample_audvis-ave.fif'
condition = ["Right visual", "Left visual","Left Auditory","Right Auditory"]
info_ = mne.io.read_info(evoked_fname)
n_channels = 305

for i in range(len(condition)):
    evoked = mne.read_evokeds(evoked_fname, condition=condition[i], baseline=(None, 0))
    evoked = evoked.pick_types(meg = True)
    data = evoked.get_data()
    
    match i:
        case 0:
            data = data[:,160:205]
            data_new = np.zeros((n_channels,int(data.shape[1]/5)))
            for j in range(int(data.shape[1]/5)):
                data_new[:,j] = np.mean(data[:,j*5:j*5+5],axis = 1)
        case 1:
            data = data[:,180:265]
            data_new = np.zeros((n_channels,int(data.shape[1]/5)))
            for j in range(int(data.shape[1]/5)):
                data_new[:,j] = np.mean(data[:,j*5:j*5+5],axis = 1)
        case 2:
            data = data[:,150:215]
            data_new = np.zeros((n_channels,int(data.shape[1]/5)))
            for j in range(int(data.shape[1]/5)):
                data_new[:,j] = np.mean(data[:,j*5:j*5+5],axis = 1)
        case 3:
            data = data[:,155:215]
            data_new = np.zeros((n_channels,int(data.shape[1]/5)))
            for j in range(int(data.shape[1]/5)):
                data_new[:,j] = np.mean(data[:,j*5:j*5+5],axis = 1)
                
    np.save('C:/Users/User/Desktop/Project/data/'+condition[i], np.array(data_new))
        
