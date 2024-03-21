import util.Convertor
from datasets.pad_events_threshold import Preprocessor
import argparse


'''
Function to convert pt file (point cloud based) to hdf5 file (image based)
args:
  fin: input file in .pt format
  fout: output file in .h5 format
  preprocessor: preprocessor file in .pkl format
'''

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fin', type=str)
  parser.add_argument('--fout', type=str)
  parser.add_argument('--preprocessor', type=str, default='datasets/test/dataset_2_padded_nentry1129To1269_preprocessor.pkl')
  args = parser.parse_args()

  Converter_ = util.Convertor.Convertor(args.fin, 0.0, preprocessor=args.preprocessor)
  Converter_.invert(-99)
  Converter_.digitize()
  Converter_.to_h5py(args.fout)
