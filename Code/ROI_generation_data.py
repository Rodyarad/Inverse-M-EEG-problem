import numpy as np
import mne
from mne.datasets import sample

# Import sample data
data_path = sample.data_path()
subject = "sample"
sample_dir = data_path / "MEG" / "sample"
raw_fname = sample_dir / "sample_audvis_raw.fif"
subjects_dir = data_path / "subjects"
trans = sample_dir / "sample_audvis_raw-trans.fif"
evoked_fname = data_path / 'MEG' / subject / 'sample_audvis-ave.fif'
raw_empty_room_fname = data_path / "MEG" / "sample" / "ernoise_raw.fif"

# raw_empty_room = mne.io.read_raw_fif(raw_empty_room_fname) Recording the signal without the patient to create more
# truthful noise

# the number of clock cycles of the modeled signal and the number of examples where a particular area in the brain is
# the triggering response
time_counts = 17
num_samples_for_ROI = 100

# Dictionary of areas in the brain and their neighboring areas
neighbour_lh = np.load('neighbour_lh.npy', allow_pickle=True).item()

info = mne.io.read_info(evoked_fname)

conductivity = (0.3,)  # for single layer

# BEM model ( describes the geometry of the head the conductivities of the different tissues)
model = mne.make_bem_model(
    subject="sample", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)

bem = mne.make_bem_solution(model)

# setup source space
src = mne.setup_source_space(
    subject, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir
)

# the forward operator computation
fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)

# correction the forward operator
ch_names = info['ch_names'][:306].copy()
ch_names.remove('MEG 2443')

fwd_fixed = fwd.pick_channels(ch_names)

fwd_fixed = mne.convert_forward_solution(
    fwd_fixed, surf_ori=True, force_fixed=True, use_cps=True
)

# load and divide the brain areas into smaller ones
labels = mne.read_labels_from_annot(
    subject='sample', parc='aparc.a2009s', hemi="lh", subjects_dir=subjects_dir
)

new = []
for i in range(len(labels)):
    if len(labels[i].vertices) <= 1200:
        new.append(labels[i].split(parts=3, subject='sample', subjects_dir=subjects_dir))
    elif len(labels[i].vertices) >= 4300:
        new.append(labels[i].split(parts=7, subject='sample', subjects_dir=subjects_dir))
    elif 3100 >= len(labels[i].vertices) >= 1100:
        new.append(labels[i].split(parts=3, subject='sample', subjects_dir=subjects_dir))
    else:
        new.append(labels[i].split(parts=5, subject='sample', subjects_dir=subjects_dir))

new_labels = []
for i in range(len(new)):
    for j in range(len(new[i])):
        new_labels.append(new[i][j])

label_names = [label.name for label in new_labels]
n_labels = len(label_names)

# creating a left- and right-hemisphere dictionary
hemi_to_ind = {'lh': 0, 'rh': 1}
used_verts = {}
grid = {'lh': [], 'rh': []}
for i, label in enumerate(new_labels):
    surf_vertices = fwd['src'][hemi_to_ind[label.hemi]]['vertno']
    restrict_verts = np.intersect1d(surf_vertices, label.vertices)
    used_verts[label.name] = restrict_verts
    if label.name[-2:] == 'lh':
        grid['lh'].extend(restrict_verts)
    else:
        grid['rh'].extend(restrict_verts)

grid['lh'] = sorted(grid['lh'])


# grid['rh'] = sorted(grid['rh'])


# function for creating chains of area activations
def make_activations(tact, number_of_cycles, region, result, y):
    if tact == 0:
        y_tact = np.zeros((261,))
        y_tact[label_names.index(region)] = 1
    else:
        y_tact = y[tact - 1].copy()
        y_tact[label_names.index(region)] = 1
    y.append(y_tact)

    activated_labs_ind = y[tact].nonzero()[0]

    for index in range(len(activated_labs_ind)):
        inv_tact = tact - 1
        time_act = 0
        while inv_tact != -1:
            if y[inv_tact][activated_labs_ind[index]] == 1:
                time_act += 1
            if y[inv_tact][activated_labs_ind[index]] != 1 and inv_tact == tact - 1:
                break
            inv_tact -= 1
        if time_act == 4:
            y[tact][activated_labs_ind[index]] = 0

    lab_id = 1 if region.endswith('lh') else 0

    dip_vector = np.zeros((7498,))

    if lab_id:
        neighbour_regions = neighbour_lh[region]
        p = [1 / len(neighbour_regions) for i in range(len(neighbour_regions))]

        activated_labs_ind = y[tact].nonzero()[0]
        activated_labs = np.array(label_names)[activated_labs_ind]
        for region in activated_labs:
            for index in range(len(used_verts[region])):
                dip_vector[grid['lh'].index(used_verts[region][index])] = 1

        result.append(dip_vector)

    '''
    else:
      neighbour_regions = neighbour_rh[region]
      p = [1/len(neighbour_regions) for i in range(len(neighbour_regions))]

      activated_labs_ind = y[tact].nonzero()[0]
      activated_labs = np.array(label_names)[activated_labs_ind]
      for region in activated_labs:
        for i in range(len(used_verts[region])):
          dip_vector[len(grid['lh']) + grid['rh'].index(used_verts[region][i])] = 1

      result.append(dip_vector)
    '''

    r = np.random.uniform(0.0, 1.0)
    p.insert(0, 0)
    p = np.cumsum(p)
    for h in range(len(p)):
        if r >= p[h]:
            p[h] = 1
        else:
            p[h] = 0
    i_neigh = sum(p) - 1

    if tact == number_of_cycles - 1:
        return
    else:
        tact = tact + 1

    make_activations(tact, number_of_cycles, neighbour_regions[int(i_neigh)], result, y)


# function for adding noise
def make_noise(sig, time_counts, num_sensor, target_snr_db):
    rez = np.zeros((num_sensor, time_counts))
    for i in range(num_sensor):
        watts = sig[i][:] ** 2

        sig_avg_watts = np.mean(watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)

        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)

        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(watts))

        rez[i] = sig[i] + noise_volts
    return rez


# forward operator
leadfield = fwd_fixed["sol"]["data"]

# noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)

# X - examples, Y - target
X = []
Y = []

# Activity generation
count_lh = 0
for lab_key in neighbour_lh.keys():
    for samples in range(num_samples_for_ROI):
        result_sample = []
        Y_sample = []
        make_activations(0, time_counts, lab_key, result_sample, Y_sample)
        result_sample = np.array(result_sample)
        result_sample = result_sample * 1e-8
        Y_sample = np.array(Y_sample)
        X_sample = np.dot(leadfield, result_sample.T)
        X_sample = make_noise(sig=X_sample, time_counts=time_counts, num_sensor=305, target_snr_db=20)
        X.append(X_sample.T)
        Y.append(Y_sample)
        print('Сгенерирована последовательность №', count_lh)
        # np.save('C:/Users/User/Desktop/project/data/X_train_' + str(count_lh), X_sample)
        # np.save('C:/Users/User/Desktop/project/data/Y_train_' + str(count_lh), Y_sample)
        count_lh += 1

'''
count_rh = 0
for lab_key in neighbour_rh.keys():
    for samples in range(num_samples_for_ROI):
        result_sample = []
        Y_sample = []
        make_activations(0,time_counts, lab_key, result_sample, Y_sample)
        result_sample = np.array(result_sample)
        result_sample = result_sample * 1e-9
        Y_sample = np.array(Y_sample)
        X_sample = np.dot(leadfield, result_sample.T)
        X_sample = make_noise(sig = X, time_counts= time_counts, num_sensor = 305, target_snr_db=20)
        X.append(X_sample)
        Y.append(Y_sample)
        # np.save('C:/Users/User/Desktop/project/data/X_train_' + str(count_lh), X_sample)
        # np.save('C:/Users/User/Desktop/project/data/Y_train_' + str(count_lh), Y_sample)
        count_rh+=1
'''

np.save('C:/Users/User/Desktop/Project/data/X_train_dop', np.array(X))
np.save('C:/Users/User/Desktop/Project/data/Y_train_dop', np.array(Y))
