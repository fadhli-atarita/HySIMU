#! /bin/bash
#SBATCH --qos=privileged
#SBATCH --job-name=project_name
#SBATCH --mail-user=your@email
#SBATCH --mail-type=ALL
#SBATCH --error=error_%x.log
#SBATCH --output=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}\
/set_matlab_runtime_path
/ttps://www.mathworks.com/help/compiler/mcr-path-settings-for-run-time-deployment.html"

SCRIPTS_DIR="/path/to/hysimu_dir"
INPUT_FILE="hysimu_input_template.xlsx"

module load load_all_modules
source activate_your_python_env

python "$SCRIPTS_DIR/hysimu_main.py" "$INPUT_FILE"
