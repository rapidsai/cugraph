# source this script (ie. do not run it) for easier job control from the shell
# FIXME: change this and/or GaaS so PYTHONPATH is not needed
PYTHONPATH=/Projects/GaaS/python python client1_script.py &
sleep 1
PYTHONPATH=/Projects/GaaS/python python client2_script.py &
PYTHONPATH=/Projects/GaaS/python python client2_script.py &
PYTHONPATH=/Projects/GaaS/python python client2_script.py &
PYTHONPATH=/Projects/GaaS/python python client2_script.py &
PYTHONPATH=/Projects/GaaS/python python client2_script.py &
PYTHONPATH=/Projects/GaaS/python python client2_script.py &
PYTHONPATH=/Projects/GaaS/python python client2_script.py
