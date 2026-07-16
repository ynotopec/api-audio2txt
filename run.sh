#!/bin/bash

serverAddress=$1
portNumber=$2

pythonVersion=python3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pythonDir=~/"venv/$(basename "${DIR}")"
cd $DIR

deactivate 2>/dev/null
mkdir -p "${pythonDir}"
${pythonVersion} -m venv "${pythonDir}"
source "${pythonDir}"/bin/activate

#intall
#${pythonVersion} -m pip cache purge ; ${pythonVersion} -m pip install -U pip setuptools wheel ; ${pythonVersion} -m pip install -U -r requirements.txt
#optimize space
#(jdupes -X size+:99M -r -L ~ >/dev/null 2>&1 )&

export HF_HUB_DISABLE_TELEMETRY=1
if [ ! -z "${serverAddress}" ] ;then
  export GRADIO_SERVER_NAME="${serverAddress}"
  export SERVER_NAME="${serverAddress}"
fi
if [ ! -z "${portNumber}" ] ;then
  export GRADIO_SERVER_PORT="${portNumber}"
  export SERVER_PORT="${portNumber}"
  export BACK_PORT=$((SERVER_PORT + 1))
fi
export CUDA_LAUNCH_BLOCKING=1

# Charger les variables d'environnement depuis .env
if [ -f ".env" ]; then
#  export $(grep -v '^#' .env | xargs)
  set -a
  source .env
  set +a
else
  echo ".env file not found!"
  exit 1
fi

#${pythonVersion} app.py $([ ! -z "${serverAddress}" ] && echo --host ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --port ${portNumber})
#${pythonVersion} -m streamlit run app.py --browser.gatherUsageStats false $([ ! -z "${serverAddress}" ] && echo --server.address ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --server.port ${portNumber})
${pythonVersion} -m uvicorn app:app $([ ! -z "${serverAddress}" ] && echo --host ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --port ${portNumber})
#${pythonVersion} back.py
