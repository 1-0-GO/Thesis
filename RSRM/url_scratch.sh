#!/bin/bash

# Copyright 2025 www.github.com/gh0stwin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# First Section - Defining input arguments (not really important for this tutorial)
ARGS_LONG="learners:,actors:,actor_per_device:,network:,method:,args:,env:,num_envs:,frame_stack:,noise:,stickyaction:,tests:,"
ARGS_LONG+="logs:,batch:,eval_batch:,rollout:,buffer_size:,,steps:,epochs:,unroll_epochs:,warmup:,task:,cpu:,mem:,after:"

if ! OPTIONS=$(getopt -o "" -l $ARGS_LONG -- "$@")
then
    exit 1
fi

eval set -- "$OPTIONS"

while [ $# -gt 0 ]
do
    case $1 in
    --learners) LEARNERS=${2} ; shift ;;
    --actors) ACTORS=${2} ; shift ;;
    --actor_per_device) ACTOR_PER_DEVICE=${2} ; shift ;;
    --network) NETWORK=${2} ; shift ;;
    --method) METHOD=${2} ; shift ;;
    --args) ARGS=${2} ; shift ;;
    --env) ENV=${2} ; shift ;;
    --num_envs) NUM_ENVS=${2} ; shift ;;
    --frame_stack) FRAME_STACK=${2} ; shift ;;
    --noise) NOISE=${2} ; shift ;;
    --stickyaction) STICKY_ACTION=${2} ; shift ;;
    --tests) TESTS=${2} ; shift ;;
    --logs) LOGS=${2} ; shift ;;
    --batch) BATCH=${2} ; shift ;;
    --eval_batch) EVAL_BATCH=${2} ; shift ;;
    --rollout) ROLLOUT=${2} ; shift ;;
    --buffer_size) BUFFER_SIZE=${2} ; shift ;;
    --steps) STEPS=${2} ; shift ;;
    --epochs) EPOCHS=${2} ; shift ;;
    --unroll_epochs) UNROLL_EPOCHS=${2} ; shift ;;
    --warmup) WARMUP=${2} ; shift ;;
    --task) TASK=${2} ; shift ;;
    --cpu) USER_CPU=${2} ; shift ;;
    --mem) USER_MEM=${2} ; shift ;;
    --gpu) USER_GPU=${2} ; shift ;;
    --after) AFTER=${2} ; shift ;;
    (--) shift; break;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
    (*) break;;
    esac
    shift
done

# If flags are not defined (not given in the command line), then give default values
if [ -z "$LEARNERS" ]; then
    LEARNERS=0
fi

if [ -z "$ACTORS" ]; then
    ACTORS=0
fi

if [ -z "$ACTOR_PER_DEVICE" ]; then
    ACTOR_PER_DEVICE=2
fi

if [ -z "$NETWORK" ]; then
    NETWORK="cnn"
fi

if [ -z "$METHOD" ]; then
    METHOD="sac"
fi

if [[ "$METHOD" = @("sac"|"metra"|"metra_state"|"metra_wm_on"|"ssd") ]]; then
    METHOD_GROUP="sac"
elif [[ "$METHOD" = @("dreamer"|"dreamer_continuous") ]]; then
    METHOD_GROUP="wm"
fi

METHOD_SCRIPT="ff_${METHOD}"
METHOD_NET="${NETWORK}_${METHOD}"

if [ -z "$ENV" ]; then
    ENV="shimmy/humanoid"
fi

if [ -z "$NUM_ENVS" ]; then
    NUM_ENVS=16
fi

if [ -z "$NOISE" ]; then
    NOISE=0
fi

if [ -z "$FRAME_STACK" ]; then
    FRAME_STACK=3
fi

if [ -z "$STICKY_ACTION" ]; then
    STICKY_ACTION=0
fi

if [ -z "$TESTS" ]; then
    TESTS=8
fi

if [ -z "$LOGS" ]; then
    LOGS=100
fi

if [ -z "$BATCH" ]; then
    BATCH="256"
fi

if [ -z "$EVAL_BATCH" ]; then
    EVAL_BATCH=24
fi

if [ -z "$ROLLOUT" ]; then
    ROLLOUT="200"
fi

if [ -z "$BUFFER_SIZE" ]; then
    BUFFER_SIZE="300000"
fi

if [ -z "$STEPS" ]; then
    STEPS=2000000
fi

if [ -z "$EPOCHS" ]; then
    EPOCHS=200
fi

if [ -z "$UNROLL_EPOCHS" ]; then
    UNROLL_EPOCHS=200
fi

if [ -z "$WARMUP" ]; then
    WARMUP=10000
fi

if [ -z "$TASK" ]; then
    TASK=run_explore
fi


if [ -z "$USER_CPU" ]; then
    USER_CPU=8
fi

if [ -z "$USER_MEM" ]; then
    USER_MEM=64
fi

SEED=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff ))
ENV_SEED=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff ))

if [[ "$ENV" ==  *"shimmy"* ]]; then
    ARGS="env.scenario.task=$TASK $ARGS"
    ARGS="env.kwargs.from_pixels=true $ARGS"
    ARGS="env.kwargs.task_kwargs.random=$ENV_SEED $ARGS"
    ARGS="env.kwargs.frame_stack.num_frames=$FRAME_STACK $ARGS"
    ARGS="env.kwargs.obs_noise_prob=$NOISE $ARGS"
    ARGS="env.kwargs.sticky_action.repeat_prob=$STICKY_ACTION $ARGS"
    ARGS="env.kwargs.sticky_action.seed=$((ENV_SEED+1)) $ARGS"
elif [[ "$ENV" ==  *"ale"* ]]; then
    ARGS="env.scenario.name=$TASK $ARGS"
    ARGS="env.kwargs.repeat_action_probability=0.25 $ARGS"
    ARGS="env.kwargs.atari_preprocess.screen_size=64 $ARGS"
    ARGS="env.kwargs.atari_preprocess.grayscale_obs=true $ARGS"
    ARGS="env.kwargs.atari_preprocess.grayscale_newaxis=true $ARGS"
    ARGS="env.kwargs.clip_reward=false $ARGS"
    ARGS="~env.kwargs.frame_concat $ARGS"
fi

# Just some logic to partition GPUs
declare -A GPUS=()

for IDX in $(echo "$LEARNERS,$ACTORS" | tr "," "\\n"); do
    if [ ${GPUS[IDX]+_} ]; then
        VALUE=$((${GPUS[IDX]}+1))
        unset GPUS[IDX]
        GPUS+=( [$IDX]=$VALUE )
    else
        VALUE=1
        GPUS+=( [$IDX]=$VALUE )
    fi
done

COUNT=0

for IDX in "${!GPUS[@]}"; do
    COUNT=$((COUNT+${GPUS[$IDX]}))
done

DEVICE_ARGS="+arch.learner.device_ids=[$LEARNERS]"
DEVICE_ARGS="+arch.actor.device_ids=[$ACTORS] $DEVICE_ARGS"
TARGET_ENT_RATIO=$(bc -l <<< 2000000/$STEPS)
TARGET_ENT_RATIO=$(printf "%.2f" $TARGET_ENT_RATIO)
UNIQUE_TOKEN=$(date +"%Y%m%d%H%M%S")

# Second Section: sbtach script as a here document
# Note that variables are normally evaluated inside,
# if `ABC=10`, then `$ABC` will be evaluated (replaced) to 10.
# If you want to read and evaluate variables only when
# sbatch runs, then you have to use `\$ABC`.
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=URL--$METHOD_SCRIPT--$ENV # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=$USER_CPU            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem="$USER_MEM"G                   # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=shard:24               # number of gpus per node
#SBATCH --export=ALL                    # export all environment variables
#SBATCH --time=96:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send mail when job begins
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH --mail-type=fail                # send mail if job fails
#SBATCH --mail-user=fabiovital@tecnico.ulisboa.pt
#SBATCH --dependency=$AFTER

set -x  # Echo commands being executed (useful for debugging)

# Unset every conda variable (just a sanity check)
unset CONDA_ENVS_PATH
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# Move $HOME to the job's folder created by slurm
export HOME=/scratch/slurm-jobs/\$SLURM_JOB_ID
chmod -R 700 \$HOME  # grous and others can't read / write / execute
cd \$HOME

# Remove any local executables (e.g., inside cfs. We are going to
# install everything from scratch)
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games

# conda needs .bashrc to be initialized without restart your shell
touch \$HOME/.bashrc
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash \$HOME/Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3  # Install without any prompts
export PATH=\$HOME/miniconda3/bin:\$PATH  # Add conda to path

source \$HOME/.bashrc  # run `conda init`
conda init bash  # run `conda init`, needed?
source \$HOME/.bashrc  # run `conda init`, needed?
conda update -n base -c defaults conda -y  # Update conda

# Copy project (already in cfs) and install conda env from `environment.yml`
cp -r /cfs/home/u035701/dev/RSRM \$HOME/RSRM
cd \$HOME/RSRM
conda env create --file environment.yml
export USER_CONDA_ENV=RSRM  # conda env name
conda activate \$USER_CONDA_ENV

# Run experiment script
python main.py --task GDM --num_test 1 --json_path config/ecology.json --threshold 1e-10 --fit A,T --split Archipelago,species


mkdir -p /cfs/home/u035701/out/$UNIQUE_TOKEN
cp -r output/* /cfs/home/u035701/out/$UNIQUE_TOKEN
set +x

EOF