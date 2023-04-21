import torch, os, sys
import numpy as np
import torch.nn.functional as F
sys.path.insert(1, '../')
import utils

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    filename = 'dataset_2_1_graph_0.pt'
    loaded_file = torch.load(filename)
    showers = loaded_file[0]
    print(f'list of {len(showers)} showers of type {type(showers[0])} from file {filename}')
    incident_energies = loaded_file[1]
    max_nhits = 0
    custom_data = utils.cloud_dataset(filename, transform=utils.rescale_energies(), transform_y=utils.rescale_conditional(), device=device)

    # Have to homogenise with data padding before using dataloaders
    padded_showers = []
    rescaled_incident_e = []
    for shower in custom_data.data:
        if len(shower.x) > max_nhits:
            max_nhits = len(shower.x)
    shower_count = 0
    for showers in custom_data.data:
        pad_hits = max_nhits-len(showers.x)
        rescaler = utils.rescale_energies()
        shower_data_transformed = rescaler(showers.x,incident_energies[shower_count])
        padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=-20)
        padded_showers.append(padded_shower)
        shower_count+=1
    
    for incident_e in incident_energies:
        rescaler_y = utils.rescale_conditional()
        incident_e_transformed = rescaler_y(incident_e,custom_data.min_y,custom_data.max_y)
        rescaled_incident_e.append(incident_e_transformed)

    print('padded_showers: ', padded_showers)
    print('rescaled_incident_e: ', rescaled_incident_e)
    print(f'list of {len(padded_showers)} padded_showers of type {type(padded_showers[0])}')
    print(f'list of {len(rescaled_incident_e)} padded_showers of type {type(rescaled_incident_e[0])}')
    torch.save([padded_showers,rescaled_incident_e], 'padded_dataset_2_1_graph_0.pt')

if __name__=='__main__':
    main()
    padded_filename = 'padded_dataset_2_1_graph_0.pt'
    padded_loaded_file = torch.load(padded_filename)
    padded_showers = padded_loaded_file[0]
    print(f'list of {len(padded_showers)} showers of type {type(padded_showers[0])} from file {padded_filename}')
    incident_energies = padded_loaded_file[1]