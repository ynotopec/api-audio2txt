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
#${pythonVersion} -m pip cache purge
#${pythonVersion} -m pip install -U pip
#${pythonVersion} -m pip install -U -r requirements.txt
#optimize space
#(jdupes -X size+:99M -r -L ~ >/dev/null 2>&1 )&

export OPENAI_API_MODEL="chat-leger"
export OPENAI_API_BASE="https://api-ai.numerique-interieur.com/v1"
export OPENAI_API_KEY="sk-i23-n5W-kyD"

if [ ! -z "${serverAddress}" ] ;then
  export GRADIO_SERVER_NAME="${serverAddress}"
  export SERVER_NAME="${serverAddress}"
fi
if [ ! -z "${portNumber}" ] ;then
  export GRADIO_SERVER_PORT="${portNumber}"
  export SERVER_PORT="${portNumber}"
fi
export CUDA_LAUNCH_BLOCKING=1

#${pythonVersion} app.py
#${pythonVersion} -m streamlit run app.py --browser.gatherUsageStats false $([ ! -z "${serverAddress}" ] && echo --server.address ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --server.port ${portNumber})
${pythonVersion} -m uvicorn app:app --reload $([ ! -z "${serverAddress}" ] && echo --host ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --port ${portNumber})
