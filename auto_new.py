import re
import sys
import os
import time
import json
import argparse
import pickle
from subprocess import call, Popen, PIPE
import csv
from threading import Thread
from fabric.api import run
from concurrent.futures import ThreadPoolExecutor


def wake_hosts(macs_list, cols, host_nums=None):
    systems = []
    macs = []
    sys_nums = []

    with open(macs_list, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[cols[0]] and row[cols[1]] and row[3] != 'v':
                systems.append(row)
                macs.append(row[cols[0]])
                sys_nums.append(int(row[cols[1]]))

    statuses = []
    print(list(zip(sys_nums, macs)), host_nums)
    if host_nums:
        for i, m in enumerate(macs):
            if sys_nums[i] in host_nums:
                print(sys_nums[i])
                statuses.append(call(["wakeonlan", m.replace(' ', '')]))
    else:
        for i in macs:
            statuses.append(call(["wakeonlan", i]))

    return statuses


def get_hosts(macs_list, cols):
    # Generate the hosts here
    # Basically load the mac addresses from the file and find the ip addresses
    # Return all the hosts which are awake
    # Uses, ip addr range 10.5.0.0/23, TODO: parametrize it.

    systems = []
    macs = []
    sys_nums = []
    with open(macs_list, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[cols[0]] and row[cols[1]]:
                systems.append([row[cols[0]].strip(), int(row[cols[1]])])
                macs.append(row[cols[0]].strip())
                sys_nums.append(int(row[cols[1]]))

    meh = dict(systems)

    p = Popen(['arp-scan', '10.5.0.0/23'], stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    out = output.decode('utf-8').split('\n')

    hosts = []
    for i in out:
        if i.find('\t') != -1:
            hosts.append([i.split('\t')[j] for j in [0, 1]])

    hosts_alive = []

    for i in hosts:
        if i[1] in meh.keys():
            hosts_alive.append(i + [meh[i[1]]])
    hosts_alive.sort(key=lambda x: int(x[2]))

    total_hosts_not_alive = range(1, 68)
    bleh = []
    for i in hosts_alive:
        bleh.append(int(i[2]))

    total_hosts_not_alive = set(total_hosts_not_alive) - set(bleh)
    hosts_not_alive = set(sys_nums) - set(bleh)

    return hosts_alive, hosts_not_alive, total_hosts_not_alive


def get_ips(hosts):
    host_ips = [i[0] for i in hosts]
    return host_ips


def check_ssh(ips):
    for ip in ips:
        pass  # check port 22 with run("telnet ip 22")


def create_fabstring(host=None, user='user', password='password'):
    if host:
        fab_string = "from fabric.api import *\n"
        fab_string += "env.hosts = " + host + "\n"
        fab_string += "env.user = '" + user + "'\n"
        fab_string += "env.password = '" + password + "'\n"
        fab_string += "env.reject_unknown_hosts = False\n"
        fab_string += "env.warn_only = True\n\n\n"
        return fab_string
    else:
        print("Host not found. The program will exit")
        sys.exit()


# For each host launch a separate thread which will write its
# own temp_host.py file and execute the script for each host
# checking the output and error streams. For anything that's
# there in the error stream, it can be parsed incrementally
# over time.
# BUG: paramiko apparently doesn't check if the password is incorrect
#          and keeps trying to connect infinitely. Must be careful.
#          I think fabric has to pass the number of retries to paramiko
#          And it doesn't handle incorrect password. Yay!
def run_func(func_name, host_num, rt_dict, user_pass, host, fs):
    # A fabstring is created here for each function. This is not
    # really desired behaviour.
    #
    # For each host start a separate subprocess and store
    # the output. This code however will not do
    # splitting on functions. Finding func and then reconstructing.
    username, password = user_pass.strip().split(',')

    funcs_list = fs.split('def')
    func_string = 'def' + [i for i in funcs_list if i.startswith(" " + func_name)][0] + "\n    run('exit')"

    print("Running " + func_name + " on " + host)
    fabstring = create_fabstring('[\'' + host + '\']', username, password)
    fabstring += func_string
    fname = 'temp_' + host.replace('.', '_') + '.py'
    with open(fname, 'w') as f:
        f.write(fabstring)

    p = Popen(['fab', '-D', '-f', fname, func_name], stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    out = output.decode('utf-8')
    err = err.decode('utf-8')
    rc = p.returncode
    # store in a dict and put in separate locations
    rt_dict[host_num] = [host, out, err, rc]
    os.remove(fname)
    if os.path.exists(fname + 'c'):
        os.remove(fname + 'c')

    return


def get_hosts_ips(macs_file, cols):
    hosts = get_hosts(macs_file, cols)
    awake_host_nums = [h[2] for h in hosts[0]]
    ips = get_ips(hosts[0])
    hosts_ips = dict([(i[0], i[2]) for i in hosts[0]])
    return ips, awake_host_nums, hosts_ips


def run_on_all(func, fs, host_nums, ips, rt, user_password, threaded, timeout):
    rt = {}
    if threaded:
        thread_list = [Thread(target=run_func, args=(func, host_num, rt, user_password),
                              kwargs={'host': ip, 'fs': fs})
                       for host_num, ip in zip(host_nums, ips)]

        for thread in thread_list:
            thread.start()
        ### DEBUG ###
        print("Started threads")
        print("Function is " + func)
        ### END DEBUG ###    

        for thread in thread_list:
            thread.join(timeout=timeout)

        ### DEBUG ###
        print("Joined threads")
        ### END DEBUG ###    
    else:
        for host_num, ip in zip(host_nums, ips):
            run_func(func, host_num, rt, user_password, host=ip, fs=fs)

    return rt


# The function currently is very raw
# Need to make classes and abstract everything also
def install_package_centralized(package, ips, timeout=60):
    fs = """
def get_uris():
    sudo('apt-get install --reinstall --print-uris -y ' + package)

def install_stuff():
    put('temp /tmp/fab_tmp')
    sudo('rm var/lib/dpkg/lock')
    sudo('dpkg -i /tmp/fab_tmp/*.deb')
    sudo('rm /tmp/fab_tmp')
"""
    rt = run_on_all('get_uris', fs, ips, 10)
    pat = re.compile('http.*?deb')
    urls = [set(pat.findall(v[0])) for v in rt.values()]
    urls = set.union(*urls)
    # with the assumption that all the ips are alive and accessible
    # And that the current system is synced with the rest w.r.t. packages
    download_stuff(urls)
    run_on_all('install_stuff', fs, ips, timeout=timeout)


def download_stuff(urls):
    with ThreadPoolExecutor(max_workers=16) as t:
        for url in urls:
            command = "until test $? -eq 0; do wget  -c --directory-prefix=temp" + url + "; done"
            t.submit(call, ['bash', '-c', command])


def do_stuff(macs_file, cols, funcs,
             host_nums, hosts_ips, ips, run_funcs, user_password, threaded, timeout=60):
    ips_hosts = dict([(v, k) for k, v in hosts_ips.items()])
    ips = [ip for ip in ips if hosts_ips[ip] in host_nums]

    ### DEBUG ###
    print("inside do_stuff running on " + str(ips))
    ### END DEBUG ###

    fs = funcs
    fs += '''
def test():
    # put("filename_wpath", "/home/rootroot")
    # sudo("cd /home/rootroot/; dpkg -i *.deb")
    # run("echo 'export PATH=PATH:some_path' > /home/exam/.bashrc")
    # run("which gprolog")
    # put("/home/rootroot/gprolog*.deb", "/home/rootroot/")
    # run("cd /home/rootroot")
    # sudo("dpkg -i gprolog*deb")
    # sudo("iptables something")
    # check numpy, scipy, matplotlib
    # run('which python &> /dev/null; echo "import numpy, scipy, matplotlib" | python - ')
    # put('/home/joe/libindicator7_12.10.2+16.04.20151208-0ubuntu1_amd64.deb', '/home/rootroot')
    # put('/home/joe/libappindicator1_12.10.1+16.04.20170215-0ubuntu1_amd64.deb', '/home/rootroot')
    # put('/home/joe/google-chrome-stable_current_amd64.deb', '/home/rootroot')
    # sudo('rm var/lib/dpkg/lock')
    # sudo('dpkg -i /home/rootroot/lib*indicator*deb')
    # sudo('dpkg -i /home/rootroot/google-chrome-stable_current_amd64.deb')
    put("apt_temp", "/home/rootroot/")
    sudo('dpkg --remove appstream gnome-software ubuntu-software')
    sudo("dpkg -i /home/rootroot/apt_temp/*.deb")
    sudo("apt-get update")
    sudo("apt-get -f -y install")

def check_chrome():
    run('which google-chrome')
    '''
    rt_list = []
    for func in run_funcs:
        rt = {}
        rt_list.append(run_on_all(func, fs, host_nums, ips, rt, user_password, threaded, timeout))

    ### DEBUG ###
    print("Ran all")
    ### END DEBUG ###    

    for i, rt in enumerate(rt_list):
        for k, v in rt.items():
            # if v[1]:
            #     print("sys number " + str(ips_hosts[k]) + " with mac " + k + " returned some fabric error for function " + run_funcs[i])
            if v[2]:
                print("sys number " + str(k) + " with ip addr " + ips_hosts[k] + " returned some internal error for function " + run_funcs[i])
                print(v[2])

    return hosts_ips, rt_list

def main():
    parser = argparse.ArgumentParser(
        description='''
        Fabric remote systems manager.
        Usage: fabric --macs-file (-m) macs_file --columns (-c) 1,2
        where macs_file is a CSV file that contains the macs and the
        system numbers.  By default two columns are read. system_macs
        and system_numbers.  The columns in the macs file can be
        specified via --columns.  Rest of the configuration has to be
        there in the config file (feature to be added later).  '''
    )
    parser.add_argument('--config-file', type=str,
                        help='A config file which may contain the values to options')
    parser.add_argument('--macs-file', '-m', type=str, default='all_macs_sorted.csv',
                        help='the csv file containing the mac addresses')
    parser.add_argument('--columns', '-c', type=str, default='1,2',
                        help='Read from this config file for the application')
    parser.add_argument('--funcs-file', '-f', type=str, default='',
                        help='A file containing the function definitions')
    parser.add_argument('--host-nums', '-n', type=str, default='',
                        help='Comma separated numbers of systems')
    parser.add_argument('--wake-hosts', '-w', type=bool, default=False,
                        help='Wake the hosts?')
    parser.add_argument('--print-hosts', '-p', type=bool, default=False,
                        help='Wake the hosts?')
    parser.add_argument('--run-funcs', '-r', type=str, default=False,
                        help='Comma separated list of functions to run')
    parser.add_argument('--timeout', '-t', type=int, default=60,
                        help='Comma separated list of functions to run')
    parser.add_argument('--user', type=str, default='',
                        help='Comma separated list of user/password e.g. root,r00tme')
    parser.add_argument('--threaded', type=bool, default=True,
                        help='execute each function for each host in a separate thread')
    args = parser.parse_args()

    cols = list(map(int, args.columns.split(',')))
    if not args.macs_file:
        print("No macs file specified. The program will exit.")
        sys.exit()

    if args.wake_hosts:
        print("Sending wake packets and waiting...")
        wake_hosts(args.macs_file, cols, args.host_nums if args.host_nums else None)
        time.sleep(60)

    ips, awake_host_nums, hosts_ips = get_hosts_ips(args.macs_file, cols)
    ips_hosts = dict([(v, k) for k, v in hosts_ips.items()])
    if args.print_hosts:
        print("Total " + str(len(awake_host_nums)) + " systems are awake\n", awake_host_nums)
        print("With ip addresses " + str(hosts_ips))

    if args.host_nums:
        host_nums = args.host_nums.split(',')
        host_nums = [int(h.strip()) for h in host_nums]
        if set(host_nums) - set(awake_host_nums):
            print(str(set(awake_host_nums) - set(host_nums))
                  + " hosts are not awake")
        host_nums = set.intersection(set(host_nums), set(awake_host_nums))
    else:
        host_nums = awake_host_nums

    if args.funcs_file:
        with open(args.funcs_file, 'r') as f:
            funcs = f.read()
    else:
        funcs = ''

    if args.run_funcs:
        run_funcs = [r.strip() for r in args.run_funcs.split(',')]
    else:
        run_funcs = None
    
    if run_funcs:
        print("Trying to run on ", host_nums)
        print("With ips ", [ips_hosts[k] for k in host_nums])
        print(len(awake_host_nums), " systems are online")
        hi, rt = do_stuff(args.macs_file, cols, funcs,
                          host_nums, hosts_ips, ips, run_funcs, args.user,
                          args.threaded, timeout=args.timeout)

        ### DEBUG ###
        print("Did stuff")
        ### END DEBUG ###    

        # with open('rt.pkl', 'wb') as f:
        #     pickle.dump(rt, f)
        with open('rt.json', 'w') as f:
            json.dump(rt, f)

    ### DEBUG ###
    print("Why the fuck is it not exiting?")
    ### END DEBUG ###    

    return

if __name__ == "__main__":
    main()
