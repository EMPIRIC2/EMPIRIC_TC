# TropicalCycloneAI

## Installation

1. Install Python if you don’t already have it. I am using python version 3.11.9. I would recommend using the this version because there are some issues with using the most recent Python release. You can find the installers here: https://www.python.org/downloads/release/python-3119/
2. Install git: https://git-scm.com/download/win for windows or https://git-scm.com/download/mac for mac. On windows, if it has an option in the install like “Use git in Windows command prompt”, select yes. (I haven’t used git on windows)
3. Get the code locally by opening the command line and running: `git clone https://github.com/EMPIRIC2/EMPIRIC-AI-emulation` 
    1. You’ll need to do some authentication for your github account, you can do this by making a personal access token through your github account here: https://github.com/settings/tokens. There is also a guide on this here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    2. this will put the project folder in your current working directory, so make sure to move to the directory you want to use in the command line
4.  Move into the code folder: `cd EMPIRIC-AI-emulation` 
5. Set up the virtual environment. Virtual environments are a way of managing project dependencies. There is a guide to them here https://docs.python.org/3/library/venv.html. To set one up:
    1. Create a new virtual environment: `python3 -m venv ml_env` 
    2. Activate the virtual environment: `source ml_env/bin/activate` 
    3. Install the required python packages: `pip install -r ml_requirements.txt`
6. Now you are ready to use the code!

## Running the Model

The one model on the github is at `models/unet_mean_1713754646.2664263.keras`