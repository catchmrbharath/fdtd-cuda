from structure import *
import re
from geometry import *
from matplotlib.pyplot import imshow, show

def parse_start_parameters(parameter_string):
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
    geometry = []
    while len(tokens):
        if(tokens[2] == 'LINE'):
            temp = Line(tokens[1], tokens[0], tokens[3], tokens[4], tokens[5], tokens[6])
            geometry.append(temp)
            tokens = tokens[7:]
        elif(tokens[2] == 'CIRCLE'):
            temp = Circle(tokens[1], tokens[0], tokens[3], tokens[4], tokens[5], tokens[6])
            geometry.append(temp)
            tokens = tokens[7:]
        elif(tokens[2] == 'Ellipse'):
            temp = Ellipse(tokens[1], tokens[0], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7])
            geometry.append(temp)
            tokens = tokens[8:]
        else:
            print tokens
            raise ValueError("Geometry could not be parsed")
    return geometry




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
        structure.parameters = parse_start_parameters(start_parameters)
        structure.create_dimensions()
        structure.create_arrays()

        #Match the geometry
        regex = re.compile(r'START_GEOMETRY(.*?)END_GEOMETRY')
        geometry_match = regex.search(parameters)
        geometry = []
        if geometry_match:
            geometry_parameters = geometry_match.group(1)
            geometry = geometry_parser(geometry_parameters)

        for geo in geometry:
            if geo.array_type == 'MU':
                geo.apply(structure.mu_array)
            elif geo.array_type == 'EPS':
                geo.apply(structure.epsilon_array)
            elif geo.array_type == 'SIGMA':
                geo.apply(structure.sigma_array)
            elif geo.array_type == 'SIGMA_STAR':
                geo.apply(structure.sigma_star_array)
        imshow(structure.mu_array)
        show()
        imshow(structure.epsilon_array)
        show()

        #Match the sources
        regex = re.compile(r'START_SOURCES(.*?)END_SOURCES')
        source_match = regex.search(parameters)
        if source_match:
            source_parameters = source_match.group(1)
            source_parser(source_parameters)
