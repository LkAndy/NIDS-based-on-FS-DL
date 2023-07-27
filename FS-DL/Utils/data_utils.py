import pandas as pd;
import tensorflow as tf;

kdd_tarin = 'data/KDDTrain+.csv'
kdd_test = 'data/KDDTest+.csv'
kdd_test_21 = 'data/KDDTest-21.csv'
unsw_nb15_train = 'data/UNSW_NB15_training-set.csv'
unsw_nb15_test = 'data/UNSW_NB15_testing-set.csv'

''' ********* Return training data and testing data **********  '''

def KDD_Train_data ():
    data = pd.read_csv(kdd_tarin,header=None);

    x = data.iloc[:, [0, 2, 4, 5]].replace({'tcp': 1, 'udp': 2, 'icmp': 3}) \
        .replace('SF', 1).replace(
        {'S0': 2, 'REJ': 3, 'RSTR': 4, 'RSTO': 5, 'S1': 6, 'SH': 7, 'S2': 8, 'RSTOS0': 9, 'S3': 10, 'OTH': 11}) \
        .replace(
        {'aol': 1, 'auth': 2, 'bgp': 3, 'courier': 4, 'csnet_ns': 5, 'ctf': 6, 'daytime': 7, 'discard': 8, 'domain': 9,
         'domain_u': 10, 'echo': 11,
         'eco_i': 12, 'ecr_i': 13, 'efs': 14, 'exec': 15, 'finger': 16, 'ftp': 17, 'ftp_data': 18, 'gopher': 19,
         'harvest': 20, 'hostnames': 21, 'http': 22,
         'http_2784': 23, 'http_443': 24, 'http_8001': 25, 'imap4': 26, 'IRC': 27, 'iso_tsap': 28, 'klogin': 29,
         'kshell': 30, 'ldap': 31, 'link': 32,
         'login': 33, 'mtp': 34, 'name': 35, 'netbios_dgm': 36, 'netbios_ns': 37, 'netbios_ssn': 38, 'netstat': 39,
         'nnsp': 40, 'nntp': 41, 'ntp_u': 42,
         'other': 43, 'pm_dump': 44, 'pop_2': 45, 'pop_3': 46, 'printer': 47, 'private': 48, 'red_i': 49,
         'remote_job': 50, 'rje': 51, 'shell': 52,
         'smtp': 53, 'sql_net': 54, 'ssh': 55, 'sunrpc': 56, 'supdup': 57, 'systat': 58, 'telnet': 59, 'tftp_u': 60,
         'tim_i': 61, 'time': 62, 'urh_i': 63,
         'urp_i': 64, 'uucp': 65, 'uucp_path': 66, 'vmnet': 67, 'whois': 68, 'X11': 69, 'Z39_50': 70});

    y = data.iloc[:, 41].replace('normal', 1).replace(
        ['neptune', 'ipsweep', 'satan', 'portsweep', 'smurf', 'nmap', 'back', 'teardrop',
         'warezclient', 'pod', 'guess_passwd', 'warezmaster', 'buffer_overflow', 'imap',
         'rootkit', 'multihop', 'phf', 'ftp_write', 'spy', 'loadmodule', 'land', 'perl'], 0);

    x = tf.reshape(x, (x.shape[0], 1, x.shape[1]))

    return x,y



def KDD_Test_data (data_name):
    test_data = pd.read_csv(data_name, header=None);

    test_x = test_data.iloc[:, [0, 2, 4, 5]].replace({'tcp': 1, 'udp': 2, 'icmp': 3}) \
        .replace('SF', 1).replace(
        {'S0': 2, 'REJ': 3, 'RSTR': 4, 'RSTO': 5, 'S1': 6, 'SH': 7, 'S2': 8, 'RSTOS0': 9, 'S3': 10, 'OTH': 11}) \
        .replace(
        {'aol': 1, 'auth': 2, 'bgp': 3, 'courier': 4, 'csnet_ns': 5, 'ctf': 6, 'daytime': 7, 'discard': 8, 'domain': 9,
         'domain_u': 10, 'echo': 11,
         'eco_i': 12, 'ecr_i': 13, 'efs': 14, 'exec': 15, 'finger': 16, 'ftp': 17, 'ftp_data': 18, 'gopher': 19,
         'harvest': 20, 'hostnames': 21, 'http': 22,
         'http_2784': 23, 'http_443': 24, 'http_8001': 25, 'imap4': 26, 'IRC': 27, 'iso_tsap': 28, 'klogin': 29,
         'kshell': 30, 'ldap': 31, 'link': 32,
         'login': 33, 'mtp': 34, 'name': 35, 'netbios_dgm': 36, 'netbios_ns': 37, 'netbios_ssn': 38, 'netstat': 39,
         'nnsp': 40, 'nntp': 41, 'ntp_u': 42,
         'other': 43, 'pm_dump': 44, 'pop_2': 45, 'pop_3': 46, 'printer': 47, 'private': 48, 'red_i': 49,
         'remote_job': 50, 'rje': 51, 'shell': 52,
         'smtp': 53, 'sql_net': 54, 'ssh': 55, 'sunrpc': 56, 'supdup': 57, 'systat': 58, 'telnet': 59, 'tftp_u': 60,
         'tim_i': 61, 'time': 62, 'urh_i': 63,
         'urp_i': 64, 'uucp': 65, 'uucp_path': 66, 'vmnet': 67, 'whois': 68, 'X11': 69, 'Z39_50': 70});

    test_y = test_data.iloc[:, 41].replace('normal', 1).replace(
        ['neptune', 'ipsweep', 'satan', 'portsweep', 'smurf', 'nmap', 'back', 'teardrop',
         'warezclient', 'pod', 'guess_passwd', 'warezmaster', 'buffer_overflow', 'imap',
         'rootkit', 'multihop', 'phf', 'ftp_write', 'spy', 'loadmodule', 'land',

         'mscan', 'apache2', 'processtable', 'snmpguess', 'saint', 'mailbomb',
         'snmpgetattack', 'httptunnel', 'named', 'ps',
         'sendmail', 'xterm', 'xlock', 'xsnoop', 'sqlattack', 'udpstorm',
         'worm', 'perl'], 0);

    test_x = tf.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return test_x,test_y



def UNSW_Train_data():
    data = pd.read_csv(unsw_nb15_train);

    # 3: service 11: dttl   4 :state 10： stll
    # 6:dpkts  13:dload    8:dbytes    1:dur
    x = data.iloc[:, [3, 4, 10, 11, 7, 8]].replace(
        {'-': 1, 'dhcp': 2, 'dns': 3, 'ftp': 4, 'ftp-data': 5, 'http': 6, 'irc': 7, 'pop3': 8, 'radius': 9, 'smtp': 10,
         'snmp': 11,
         'ssh': 12, 'ssl': 13}) \
        .replace({'CON': 1, 'ECO': 2, 'FIN': 3, 'INT': 4, 'no': 5, 'PAR': 6, 'REQ': 7, 'RST': 8, 'URN': 9});

    y = data.iloc[:, -1];

    x = tf.reshape(x, (x.shape[0], 1, x.shape[1]))

    return x,y



def UNSW_Test_data():
    test_data = pd.read_csv(unsw_nb15_test);

    test_x = test_data.iloc[:, [3, 4, 10, 11, 7, 8]].replace(
        {'-': 1, 'dhcp': 2, 'dns': 3, 'ftp': 4, 'ftp-data': 5, 'http': 6, 'irc': 7, 'pop3': 8, 'radius': 9, 'smtp': 10,
         'snmp': 11,
         'ssh': 12, 'ssl': 13}) \
        .replace({'CON': 1, 'ECO': 2, 'FIN': 3, 'INT': 4, 'no': 5, 'PAR': 6, 'REQ': 7, 'RST': 8, 'URN': 9, 'ACC': 10,
                  'CLO': 11});

    test_y = test_data.iloc[:, -1];

    test_x = tf.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return test_x,test_y
