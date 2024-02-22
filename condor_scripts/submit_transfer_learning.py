import os, sys
import optparse, argparse


def prepare_shell(shell_file, command, condor, FarmDir, afs_dir):

###############
# Func: prepare sh file and add it to the condor schedule.
###############

  with open(os.path.join(FarmDir, shell_file), 'w') as shell:
    shell.write('#!/bin/bash\n')
    shell.write('WORKDIR=%s\n'%afs_dir)
    #shell.write('eval `scram r -sh`\n')
    shell.write('cd ${WORKDIR}\n')
    shell.write('source script/env.sh\n')
    shell.write(command)

  condor.write('cfgFile=%s\n'%shell_file)
  condor.write('queue 1\n')


if __name__ == '__main__':
  usage = 'usage: %prog[options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--padding_value', dest='padding_value', default = 0.0, type=float)
  parser.add_argument('--dataset', dest='dataset', default = 'dataset_2_padded_nentry', type=str)
  parser.add_argument('--preproc_name', dest='preproc_name', default = 'tanh_transform', type=str)
  parser.add_argument('--keyword', dest='keyword', default = '11*.pt', type=str)
  parser.add_argument('--dataset_dir', dest='dataset_dir', default = '/eos/user/t/tihsu/SWAN_projects/ML_hackthon_transferlearning/tdsm_encoder/datasets/', type=str)
  #parser.add_argument('--working_dir', dest='working_dir', default = './', type=str)
  #parser.add_argument('--initial_model', dest='initial_model', default='initial_model.pt', type=str)
  parser.add_argument('--SDE', dest='SDE', default='VP', type=str)
  parser.add_argument('--n_epochs', dest='n_epochs', default=['1000'], type=str, nargs='+')
  parser.add_argument('--embed_dim', dest='embed_dim', default=32, type=int)
  parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
  parser.add_argument('--num_encoder_blocks', dest='num_encoder_blocks', default=8, type=int)
  parser.add_argument('--num_attn_heads', dest='num_attn_heads', default=16, type=int)
  parser.add_argument('--dropout_gen', dest='dropout_gen', default=0, type=float)
  #parser.add_argument('--cls_ffnn', action='store_true')
  #parser.add_argument('--mask_diff', action='store_true')
  parser.add_argument('--train_ratio', dest='train_ratio', default=0.9, type=float)
  parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
  parser.add_argument('--initial_lr', dest='initial_lr', default=1e-3, type=float)
  #parser.add_argument('--transfer_learning_series', dest='transfer_learning_series', default=[-0.1], nargs='+')
  #parser.add_argument('--postfix', dest='postfix', default='test', type=str)
  parser.add_argument('--sampler_steps', dest='sampler_steps', default=200, type=int)
  parser.add_argument('--n_showers_2_gen', dest='n_showers_2_gen', default=1000, type=int)
  parser.add_argument('--afs_dir', dest='afs_dir', default='/afs/cern.ch/user/t/tihsu/ML_hackathon_transferlearning', type=str)
  parser.add_argument('--JobFlavour', dest='JobFlavour', default='tomorrow', type=str)
  parser.add_argument('--serialized_model', action='store_true')
  parser.add_argument('--sampler_steps2plot', default = ['0', '50', '100', '150', '199'], type=str, nargs='+')
  parser.add_argument('--continue_', action='store_true')
  args = parser.parse_args()

  padding_value = args.padding_value
  dataset       = args.dataset
  preproc_dataset_name = args.preproc_name
  keyword       = args.keyword
  dataset_dir   = args.dataset_dir
  working_dir   = os.getcwd()
  #initial_model = args.initial_model
  SDE           = args.SDE
  embed_dim     = args.embed_dim
  hidden_dim    = args.hidden_dim
  num_encoder_blocks = args.num_encoder_blocks
  num_attn_heads= args.num_attn_heads
  dropout_gen   = args.dropout_gen
  #cls_ffnn      = args.cls_ffnn
  #mask_diff     = args.mask_diff
  train_ratio   = args.train_ratio
  batch_size    = args.batch_size
  #n_epochs      = args.n_epochs
  initial_lr    = args.initial_lr
  #transfer_learning_series = args.transfer_learning_series
  #postfix       = args.postfix
  sampler_steps = args.sampler_steps
  n_showers_2_gen = args.n_showers_2_gen
  afs_dir = args.afs_dir
  
  if args.serialized_model:
      serialized_model_command = '--serialized_model'
  else:
      serialized_model_command = ''
  continue_command = '--continue_ ' if args.continue_ else ''

  farm_dir = os.path.join(afs_dir, 'Farm')
  os.system('mkdir -p {}'.format(farm_dir))

  script_dir = os.path.join(afs_dir, 'script')
  os.system('mkdir -p {}'.format(script_dir))
  os.system('cp env.sh {}/env.sh'.format(script_dir))

  condor = open(os.path.join(farm_dir, 'condor.sub'), 'w')
  condor.write('output = %s/job_common_$(Process).out\n'%farm_dir)
  condor.write('error  = %s/job_common_$(Process).err\n'%farm_dir)
  condor.write('log    = %s/job_common_$(Process).log\n'%farm_dir)
  condor.write('executable = %s/$(cfgFile)\n'%farm_dir)
  condor.write('request_GPUs = 1\n')
#  condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
  condor.write('+JobFlavour = "%s"\n'%args.JobFlavour)
  
  for file_ in os.listdir(os.path.join(args.dataset_dir, 'tanh_transform')):
      if args.dataset not in file_: continue
      if '.pt' not in file_: continue
      if '11' not in file_: continue
      command = 'python3 {}/transfer_learning.py --working_dir {} --n_epochs {} --postfix {} --n_showers_2_gen {} --sampler_steps2plot {} {} --keyword {}*.pt {}'.format(working_dir, working_dir, ' '.join(args.n_epochs), file_.replace('.pt',''), args.n_showers_2_gen ,' '.join(args.sampler_steps2plot), serialized_model_command, file_.replace('.pt','').replace(args.dataset,''), continue_command)
      shell_file = '{}.sh'.format(file_.replace('.pt',''))
      prepare_shell(shell_file,command, condor, farm_dir, afs_dir)
  condor.close()
