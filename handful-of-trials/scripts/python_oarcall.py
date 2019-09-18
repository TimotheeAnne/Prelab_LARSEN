from subprocess import call

import os
import datetime
import argparse
import json

config_file = None
parser = argparse.ArgumentParser()
parser.add_argument("--json", 
                    help="""Path to json config file.
                     Format: [{"exp":"path/to/experiment.py", 
                     "prefix": "Prefix for result folder",
                     "result_dir":"path/to/result/dir", 
                     "walltime":"exp wall time",
                     "args": " --args1 --args 2 etc", 
                     "replicates": 3}, ...]""",
                    type=str,
                    required=True)

arguments = parser.parse_args()
if arguments.json is not None: config_file = arguments.json

with open(config_file, 'r') as f:
    config_dict_list = json.load(f)


for d in config_dict_list:
    replicates = int(d["replicates"])
    for rep_count in range(replicates):
        exp_folder = d["prefix"]
        name = d["prefix"] + "_replicate_" + str(rep_count)
        exp = d["exp"]
        args = d["args"]
        result_dir = d["result_dir"]
        walltime = d["walltime"]
        
        dir = result_dir + "/" + exp_folder + "/" + name
        script = dir + "/" + name + ".job"
        if not os.path.exists(dir):
            os.makedirs(dir)
        f = open(script, "w")  

        text = "#!/bin/bash"
        text += 'cd '+ dir
        text += '\n'
        text += 'export PATH=/home/tanne/miniconda3/bin:$PATH'
        text += '\n'
        text += '. /home/tanne/miniconda3/etc/profile.d/conda.sh'
        text += '\n'
        text += 'conda activate $1'
        text += '\n'
        text += 'shift'
        text += '\n'
        text += 'shift'
        text += '\n'
        text += 'exec python $BIN $@ -logdir /Documents/Prelab_LARSEN/handful-of-trials/scripts/log/'
        text += '\n'

        command = 'exec python -u ' + exp + " " + args
        
        print("Name: ", name)
        print("Result dir: ", result_dir)
        print("Command: ", command)
        print("") 
        
        text += command    
        f.write(text)
        f.close()

        run_command = "sh " + script
        consoleio = ' -O \"' + dir + '/%jobid%.stdout\" -E \"' + dir + '/%jobid%.stderr\"'
        print (consoleio)
        call('oarsub -n ' + name + consoleio + ' -q production -p \"cluster=\'graffiti\'\" -l gpu=1' + ',walltime=' + walltime + ' \"' + run_command + '\"', shell=True)
