#
# alma_util.py
#
# A module containing utility functions
#
import os, platform

def workdir():
    '''
    Make scripts running universally on our several machines.
    The "work" directory (supposedly containing the ALMA and smili2_dev
    directories) is different on different servers.
    On leonid2 and capelin (Lynn) it is
        /data-smili/
    On isco it is
        /data-isco/data-smili/
    On my machine it is my home directory, 
        ~ = /home/benkev
    '''
    
    hostname = platform.node()   # Host name
    hostname = hostname.split('.')[0]

    if hostname == 'isco':
        wdir = '/data-isco/data-smili/'
    elif hostname == 'leonid2' or hostname == 'capelin':
        wdir = '/data-smili/'
    else:
        wdir = os.path.expanduser('~') + '/'

    return wdir
