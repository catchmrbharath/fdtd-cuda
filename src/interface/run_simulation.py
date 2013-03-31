from structure import *
import re

def parse_start_parameters(parameter_string):
    print parameter_string
    regex = re.compile(r'(\w+)\s+=\s+(\w+)')
    parameter_list = re.findall(regex, parameter_string)
    units = {'um':1e-6,
             'cm':1e-2,
             'mm':1e-3,
             'nm':1e-9,
             'pm':1e-12,
             'ms':1e-3,
             'us':1e-6,
             'ns':1e-9,
             'ps':1e-12}
    parameter_dict = dict()
    for values in parameter_list:
        a, b = values
        if b.endswith(tuple(units.keys())):
            b = float(b[:-2]) * units[b[-2:]]
        parameter_dict[a] = b
    return parameter_dict


def geometry_parser(geometry_parameters):
    """Parses the geometry as tokens. In the
    case of bad parsing, raises ValueError """

    tokens = geometry_parameters.split()
    tokens = [t.strip() for t in tokens]


def source_parser(source_parameters): pass


def parse_file(filename):
    units_map = dict()
    units_map['um'] = 1e-6
    units_map['mm'] = 1e-3
    units_map['nm'] = 1e-9
    units_map['pm'] = 1e-12
    structure = Structure()
    string_list = []
    with open(filename) as f:
        for lines in f:
            string_list.append(lines.strip())
        parameters = " ".join(string_list)
        #Match all the default settings
        regex = re.compile(r'START_DEFAULTS(.*?)END_DEFAULTS')
        start_parameters = regex.search(parameters).group(1)
        parse_start_parameters(start_parameters)

        #Match the geometry
        regex = re.compile(r'START_GEOMETRY(.*?)END_GEOMETRY')
        geometry_match = regex.search(parameters)
        if geometry_match:
            geometry_parameters = geometry_match.group(1)
            geometry_parser(geometry_parameters)

        #Match the sources
        regex = re.compile(r'START_SOURCES(.*?)END_SOURCES')
        source_match = regex.search(parameters)
        if source_match:
            source_parameters = source_match.group(1)
                source_parser(source_parameters)
