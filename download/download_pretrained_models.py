import subprocess
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

####################
#DOWNLOAD PANN MODEL
output_dir = "./pann"
file_name = "ResNet38_mAP=0.434.pth"
dl_link = "https://zenodo.org//record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"

# Execute the wget command
command = f'wget "{dl_link}" -O {output_dir}/{file_name}'
subprocess.run(command, shell=True)

