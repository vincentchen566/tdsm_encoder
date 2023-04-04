import torch, os, sys
import torch.nn.functional as F
sys.path.insert(1, '../')
import utils
from torch_geometric.data import Data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filename = 'dataset_2_1_graph_0.pt'
    loaded_file = torch.load(filename)
    showers = loaded_file[0]
    print(f'list of {len(showers)} showers of type {type(showers[0])} from file {filename}')
    incident_energies = loaded_file[1]
    max_nhits = 0
    custom_data = utils.cloud_dataset(filename, transform=utils.rescale_energies(), device=device)
    padded_showers = []
    for shower in custom_data.data:
        if len(shower.x) > max_nhits:
            max_nhits = len(shower.x)
    
    for showers in custom_data.data:
        pad_hits = max_nhits-len(showers.x)
        padded_shower = F.pad(input = showers.x, pad=(0,0,0,pad_hits), mode='constant', value=0)
        #padded_showers.append(Data(x=padded_shower))
        padded_showers.append(padded_shower)

    print(f'list of {len(padded_showers)} padded_showers of type {type(padded_showers[0])}')
    torch.save([padded_showers,incident_energies], 'padded_dataset_2_1_graph_0.pt')

if __name__=='__main__':
    main()
    padded_filename = 'padded_dataset_2_1_graph_0.pt'
    padded_loaded_file = torch.load(padded_filename)
    padded_showers = padded_loaded_file[0]
    print(f'list of {len(padded_showers)} showers of type {type(padded_showers[0])} from file {padded_filename}')
    incident_energies = padded_loaded_file[1]