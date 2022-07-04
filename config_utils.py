
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(s, color):
    '''
    print with colors 
    '''
    if color == 'okgreen':
        return f"{bcolors.OKGREEN}{s}{bcolors.ENDC}"
    elif color == 'warning':
        return f"{bcolors.WARNING}{s}{bcolors.ENDC}"
    elif color == 'fail':
        return f"{bcolors.FAIL}{s}{bcolors.ENDC}"
    else:
        raise NotImplementedError


def print_config(config, parser=None):
    message = ''
    message += '--------------- Configures --------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        if parser is not None:
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
                comment = cprint(comment, 'warning')
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)