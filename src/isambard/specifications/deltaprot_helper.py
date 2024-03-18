import numpy as np
import json

# from utils import *
import itertools
import math
from ampal.geometry import dihedral
import os

###########################################################################################
###########################################################################################


class Deltahedron:
    def __init__(self, edge_length):
        self.edge_length = edge_length
        self.vertices = self.generate_vertices()
        self.connections = self.generate_connections()

    # @staticmethod
    def choose_deltahedron_by_rib_number(rib_num: int, edge_length: float):
        assert rib_num in [
            3,
            4,
            5,
            6,
        ], "Rib number (alpha helix count) in deltahedron can be 3,4,5,6 which corresponds to Octahedron, SnubDisophenoid, GyroSquareBipyramid, Icosahedron, according to Murzin and Finkelstein model"
        if rib_num == 3:
            return Octahedron(
                edge_length
            )  # SnubDisophenoid() GyroSquareBipyramid(Deltahedron) Icosahedron(Deltahedron)
        elif rib_num == 4:
            return SnubDisophenoid(edge_length)
        elif rib_num == 5:
            return GyroSquareBipyramid(edge_length)
        elif rib_num == 6:
            return Icosahedron(edge_length)

    def choose_deltahedron_by_name(deltahedron: str, edge_length: float):
        assert deltahedron.lower() in [
            "octahedron",
            "snub_disophenoid",
            "gyro_square_bipyramid",
            "icosahedron",
        ], f'Only {["octahedron", "snub_disophenoid", "gyro_square_bipyramid", "icosahedron"]} deltahedrons are supported.'
        if deltahedron == "octahedron":
            return Octahedron(edge_length)
        elif deltahedron == "snub_disophenoid":
            return SnubDisophenoid(edge_length)
        elif deltahedron == "gyro_square_bipyramid":
            return GyroSquareBipyramid(edge_length)
        elif deltahedron == "icosahedron":
            return Icosahedron(edge_length)


class Octahedron(Deltahedron):
    def __init__(self, edge_length):
        super(Octahedron, self).__init__(edge_length)
        self.rib_num = 3
        self.name = "octahedron"
        return

    def generate_vertices(self):
        el = self.edge_length
        xy = np.sin(np.pi / 4) * el
        return [
            (0, 0, xy),
            (0, xy, 0),
            (xy, 0, 0),
            (0, -xy, 0),
            (-xy, 0, 0),
            (0, 0, -xy),
        ]

    def generate_connections(self):
        return [
            [1, 2, 3, 4],
            [0, 2, 4, 5],
            [0, 1, 3, 5],
            [0, 2, 4, 5],
            [0, 1, 3, 5],
            [1, 2, 3, 4],
        ]


class SnubDisophenoid(Deltahedron):
    def __init__(self, edge_length):
        super().__init__(edge_length)
        self.rib_num = 4
        self.name = "snub_disophenoid"
        return

    def generate_vertices(self):
        el = self.edge_length
        # -z1 to move gh vector off x axis
        x2 = 0.644584 * el
        z1 = 0.578369 * el
        z2 = 0.989492 * el
        z3 = 1.56786 * el

        return [
            (0, el / 2, z3 - z1),  # a
            (0, -el / 2, z3 - z1),  # b
            (0, -x2, z1 - z1),  # c
            (-x2, 0, z2 - z1),  # d
            (0, x2, z1 - z1),  # e
            (x2, 0, z2 - z1),  # f
            (el / 2, 0, 0 - z1),  # g
            (-el / 2, 0, 0 - z1),  # h
        ]

    def generate_connections(self):
        return [
            [1, 3, 4, 5],
            [0, 2, 3, 5],
            [1, 3, 5, 6, 7],
            [0, 1, 2, 4, 7],
            [0, 3, 5, 6, 7],
            [0, 1, 2, 4, 6],
            [2, 4, 5, 7],
            [2, 3, 4, 6],
        ]


class GyroSquareBipyramid(Deltahedron):
    def __init__(self, edge_length):
        super().__init__(edge_length)
        self.rib_num = 5
        self.name = "gyro_square_bipyramid"
        return

    def generate_vertices(self):
        el = self.edge_length
        rl = (0.5 * el) / np.sin(np.pi / 4)
        zs = np.sin(np.pi / 3) * el
        pl = rl - (el / 2)
        z1 = np.sqrt((zs**2) - (pl**2)) / 2
        rxy = el / 2
        theta = np.arccos(rl / el)
        z2 = np.sin(theta) * el
        z3 = z1 + z2
        vertices = [
            (0, 0, z3),  # a
            (0, rl, z1),  # b
            (rl, 0, z1),  # c
            (0, -rl, z1),  # d
            (-rl, 0, z1),  # e
            (-rxy, rxy, -z1),  # f
            (rxy, rxy, -z1),  # g
            (rxy, -rxy, -z1),  # h
            (-rxy, -rxy, -z1),  # k
            (0, 0, -z3),  # l
        ]
        return vertices

    def generate_connections(self):
        return [
            [1, 2, 3, 4],
            [0, 2, 4, 5, 6],
            [0, 1, 3, 6, 7],
            [0, 2, 4, 7, 8],
            [0, 1, 3, 5, 8],
            [1, 4, 6, 8, 9],
            [1, 2, 5, 7, 9],
            [2, 3, 6, 8, 9],
            [3, 4, 5, 7, 9],
            [5, 6, 7, 8],
        ]


class Icosahedron(Deltahedron):
    def __init__(self, edge_length):
        super().__init__(edge_length)
        self.rib_num = 6
        self.name = "icosahedron"
        return

    def generate_vertices(self):
        el = self.edge_length
        rl = (el / 2) / np.sin(np.pi / 5)
        x2 = np.cos(np.pi / 2 - (2 * np.pi / 5)) * rl
        y2 = np.sin(np.pi / 2 - (2 * np.pi / 5)) * rl
        x3 = np.sin(np.pi - 2 * (2 * np.pi / 5)) * rl
        y3 = np.cos(np.pi - 2 * (2 * np.pi / 5)) * rl
        x4 = np.sin(np.pi / 5) * rl
        y4 = np.cos(np.pi / 5) * rl
        x5 = np.cos((3 * np.pi / 5) - (np.pi / 2)) * rl
        y5 = np.sin((3 * np.pi / 5) - (np.pi / 2)) * rl
        zs = np.sqrt(el**2 - (el / 2) ** 2)
        z1 = np.sqrt(zs**2 - (rl - y4) ** 2) / 2
        z2 = np.sqrt(el**2 - rl**2)
        vertices = [
            (0, 0, z1 + z2),  # a
            (x2, y2, z1),  # b
            (x3, -y3, z1),  # c
            (-x3, -y3, z1),  # d
            (-x2, y2, z1),  # e
            (0, rl, z1),  # f
            (x4, y4, -z1),  # g
            (x5, -y5, -z1),  # h
            (0, -rl, -z1),  # k
            (-x5, -y5, -z1),  # l
            (-x4, y4, -z1),  # m
            (0, 0, -z1 + -z2),  # n
        ]
        return vertices

    def generate_connections(self):
        return [
            [1, 2, 3, 4, 5],
            [0, 2, 5, 6, 7],
            [0, 1, 3, 7, 8],
            [0, 2, 4, 8, 9],
            [0, 3, 5, 9, 10],
            [0, 1, 4, 6, 10],
            [1, 5, 7, 10, 11],
            [1, 2, 6, 8, 11],
            [2, 3, 7, 9, 11],
            [3, 4, 8, 10, 11],
            [4, 5, 6, 9, 11],
            [6, 7, 8, 9, 10],
        ]


# def gen_octahedron(el):
#     """Generates vertices of an octahedron with a defined edge length.

#     Parameters
#     ----------
#     el : float
#         The edge length of the polyhedron in arbitrary units.

#     Returns
#     -------
#     vertices : [triple]
#         List containing coordinates of the vertices of the polyhedron.
#     """
#     xy = np.sin(np.pi / 4) * el
#     vertices = [
#         (0, 0, xy),  # a
#         (0, xy, 0),  # b
#         (xy, 0, 0),  # c
#         (0, -xy, 0),  # d
#         (-xy, 0, 0),  # e
#         (0, 0, -xy),  # f
#     ]
#     return vertices


# def gen_snub_disphenoid(el):
#     """Generates vertices of a snub-nosed disphenoid with a defined edge length.

#     Parameters
#     ----------
#     el : float
#         The edge length of the polyhedron in arbitrary units.

#     Returns
#     -------
#     vertices : [triple]
#         List containing coordinates of the vertices of the polyhedron.
#     """
#     # -z1 to move gh vector off x axis
#     x2 = 0.644584 * el
#     z1 = 0.578369 * el
#     z2 = 0.989492 * el
#     z3 = 1.56786 * el

#     vertices = [
#         (0, el / 2, z3 - z1),  # a
#         (0, -el / 2, z3 - z1),  # b
#         (0, -x2, z1 - z1),  # c
#         (-x2, 0, z2 - z1),  # d
#         (0, x2, z1 - z1),  # e
#         (x2, 0, z2 - z1),  # f
#         (el / 2, 0, 0 - z1),  # g
#         (-el / 2, 0, 0 - z1),  # h
#     ]
#     return vertices


# def gen_gyro_square_bipyramid(el):
#     """Generates vertices of a gyroelongated bipyramid with a defined edge length.

#     Parameters
#     ----------
#     el : float
#         The edge length of the polyhedron in arbitrary units.

#     Returns
#     -------
#     vertices : [(float, float, float)]
#         List containing coordinates of the vertices of the polyhedron.
#     """
#     rl = (0.5 * el) / np.sin(np.pi / 4)
#     zs = np.sin(np.pi / 3) * el
#     pl = rl - (el / 2)
#     z1 = np.sqrt((zs**2) - (pl**2)) / 2
#     rxy = el / 2
#     theta = np.arccos(rl / el)
#     z2 = np.sin(theta) * el
#     z3 = z1 + z2
#     vertices = [
#         (0, 0, z3),  # a
#         (0, rl, z1),  # b
#         (rl, 0, z1),  # c
#         (0, -rl, z1),  # d
#         (-rl, 0, z1),  # e
#         (-rxy, rxy, -z1),  # f
#         (rxy, rxy, -z1),  # g
#         (rxy, -rxy, -z1),  # h
#         (-rxy, -rxy, -z1),  # k
#         (0, 0, -z3),  # l
#     ]
#     return vertices


# def gen_icosahedron(el):
#     """Generates vertices of an icosahedron with a defined edge length.

#     Parameters
#     ----------
#     el : float
#         The edge length of the polyhedron in arbitrary units.

#     Returns
#     -------
#     vertices : [(float, float, float)]
#         List containing coordinates of the vertices of the polyhedron.
#     """
#     rl = (el / 2) / np.sin(np.pi / 5)
#     x2 = np.cos(np.pi / 2 - (2 * np.pi / 5)) * rl
#     y2 = np.sin(np.pi / 2 - (2 * np.pi / 5)) * rl
#     x3 = np.sin(np.pi - 2 * (2 * np.pi / 5)) * rl
#     y3 = np.cos(np.pi - 2 * (2 * np.pi / 5)) * rl
#     x4 = np.sin(np.pi / 5) * rl
#     y4 = np.cos(np.pi / 5) * rl
#     x5 = np.cos((3 * np.pi / 5) - (np.pi / 2)) * rl
#     y5 = np.sin((3 * np.pi / 5) - (np.pi / 2)) * rl
#     zs = np.sqrt(el**2 - (el / 2) ** 2)
#     z1 = np.sqrt(zs**2 - (rl - y4) ** 2) / 2
#     z2 = np.sqrt(el**2 - rl**2)
#     vertices = [
#         (0, 0, z1 + z2),  # a
#         (x2, y2, z1),  # b
#         (x3, -y3, z1),  # c
#         (-x3, -y3, z1),  # d
#         (-x2, y2, z1),  # e
#         (0, rl, z1),  # f
#         (x4, y4, -z1),  # g
#         (x5, -y5, -z1),  # h
#         (0, -rl, -z1),  # k
#         (-x5, -y5, -z1),  # l
#         (-x4, y4, -z1),  # m
#         (0, 0, -z1 + -z2),  # n
#     ]
#     return vertices


# def get_octahedron_connections():
#     return [
#         [1, 2, 3, 4],
#         [0, 2, 4, 5],
#         [0, 1, 3, 5],
#         [0, 2, 4, 5],
#         [0, 1, 3, 5],
#         [1, 2, 3, 4],
#     ]


# def get_snub_disphenoid_connections():
#     return [
#         [1, 3, 4, 5],
#         [0, 2, 3, 5],
#         [1, 3, 5, 6, 7],
#         [0, 1, 2, 4, 7],
#         [0, 3, 5, 6, 7],
#         [0, 1, 2, 4, 6],
#         [2, 4, 5, 7],
#         [2, 3, 4, 6],
#     ]


# def get_gyro_square_bipyramid_connections():
#     return [
#         [1, 2, 3, 4],
#         [0, 2, 4, 5, 6],
#         [0, 1, 3, 6, 7],
#         [0, 2, 4, 7, 8],
#         [0, 1, 3, 5, 8],
#         [1, 4, 6, 8, 9],
#         [1, 2, 5, 7, 9],
#         [2, 3, 6, 8, 9],
#         [3, 4, 5, 7, 9],
#         [5, 6, 7, 8],
#     ]


# def get_icosahedron_connections():
#     return [
#         [1, 2, 3, 4, 5],
#         [0, 2, 5, 6, 7],
#         [0, 1, 3, 7, 8],
#         [0, 2, 4, 8, 9],
#         [0, 3, 5, 9, 10],
#         [0, 1, 4, 6, 10],
#         [1, 5, 7, 10, 11],
#         [1, 2, 6, 8, 11],
#         [2, 3, 7, 9, 11],
#         [3, 4, 8, 10, 11],
#         [4, 5, 6, 9, 11],
#         [6, 7, 8, 9, 10],
#     ]


# choose_delta = {
#     3: gen_octahedron,
#     4: gen_snub_disphenoid,
#     5: gen_gyro_square_bipyramid,
#     6: gen_icosahedron,
# }

# get_connections = {
#     3: get_octahedron_connections,
#     4: get_snub_disphenoid_connections,
#     5: get_gyro_square_bipyramid_connections,
#     6: get_icosahedron_connections,
# }


def get_orientation_codes(with_dots):
    ordered_orientations_list = [
        "b3iii",
        "b3nnn",
        "b4iiiix",
        "b4iiiiy",
        "b4iiin",
        "b4inin",
        "b4innn",
        "b4nnnnx",
        "b4nnnny",
        "h4i.n",
        "l4iin",
        "l4inn",
        "b5iiiin",
        "b5iinin",
        "b5ininn",
        "b5innnn",
        "h5i.i",
        "h5n.n",
        "l5iiin",
        "l5inni",
        "l5innn",
        "l5niin",
        "b6iiniin",
        "b6ininin",
        "b6inninn",
        "h6i.i.i",
        "h6n.n.n",
        "l6innni",
        "l6niiin",
        "s6",
    ]
    if with_dots == False:
        return [i.replace(".", "_") for i in ordered_orientations_list]
    elif with_dots == True:
        return ordered_orientations_list


def get_rib_orientations():

    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the JSON file
    json_file_path = os.path.join(current_dir, "top_rib_orientations.json")

    with open(json_file_path, "r") as f:
        rib_orientations = json.load(f)

    return rib_orientations


###########################################################################################
###########################################################################################


def euclidian_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def angle_between_vectors(a, b):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(a, b)
    # Calculate the magnitudes of the vectors
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    # Calculate the cosine of the angle
    cos_angle = dot_product / (mag_a * mag_b)
    # Handle possible numerical errors due to floating point calculations
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Convert the cosine to an angle in radians
    angle_rad = np.arccos(cos_angle)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return round(angle_deg, 1)


def get_tadas_scores_for_permutation(ribs: list, deltahedron: Deltahedron):
    # A score for rib arrangement. Order of ribs matter. Direction of ribs matter. Helix axis rotation has no effect.
    # Inspired by Taylor et al.scoring.
    # Tadas score can be used to rank permutations of single chain M&F orientations.
    # Tadas score is not affected by helix rotation as looks to helices simply as ribs in space connected in sequence.

    flattened_list = [val for sublist in ribs for val in sublist]
    assert len(flattened_list) == len(
        set(flattened_list)
    ), "Helices endpoints (rib endpoints) should not be overlapping or else scores will end up infinite."

    assert (
        len(ribs) > 1
    ), "At least 2 helices (ribs) should be provided as the Tadas' score is evaluating rib interactions."

    taylor_scores = {
        "compared_helix_indexes_ij": [],
        "numeric_packing_descriptors": [],
        "angles_between_rib_vectors": [],
        "chothia_omega_angles": [],
        "taylor_letter_packing_descriptors": [],
        "distance_scores": [],
        "dihedral_angles": [],
        "orientation_scores": [],
        "sequence_proximity": [],
        "combined_helix_pair_scores": [],
    }
    # TODO can change to itertools generators.
    # for ittertools.combinations(ribs) or similar
    for i_helix_index in range(len(ribs)):
        for j_helix_index in range(len(ribs)):
            if i_helix_index >= j_helix_index:
                continue
            # compares each pair, except with oneself and ignore order of helices.

            # define the index and coordinates of ribs i and j:
            i_n_index = ribs[i_helix_index][0]  # helix i, N terminus
            i_c_index = ribs[i_helix_index][1]  # helix i, C terminus
            j_n_index = ribs[j_helix_index][0]  # helix j, N terminus
            j_c_index = ribs[j_helix_index][1]  # helix j, C terminus

            i_n_coords = np.array(deltahedron.vertices[i_n_index])
            i_c_coords = np.array(deltahedron.vertices[i_c_index])
            j_n_coords = np.array(deltahedron.vertices[j_n_index])
            j_c_coords = np.array(deltahedron.vertices[j_c_index])

            i_vector = i_c_coords - i_n_coords
            j_vector = j_c_coords - j_n_coords

            taylor_scores["compared_helix_indexes_ij"].append(
                [i_helix_index, j_helix_index]
            )

            # Getting numeric packing descriptors: nn, nc, cn, cc distances will be presented as 1121 for example.
            numeric_descriptor = get_taylor_numeric_descriptor(
                i_n_index, i_c_index, j_n_index, j_c_index, deltahedron
            )
            taylor_scores["numeric_packing_descriptors"].append(numeric_descriptor)

            angle_between_rib_vectors = angle_between_vectors(i_vector, j_vector)
            taylor_scores["angles_between_rib_vectors"].append(
                angle_between_rib_vectors
            )

            # Getting chirality: Ω from Chothia et al 1977 Structure of proteins: packing of alpha-helices and pleated sheets
            omega_angle = get_chothia_omega_angle_between_two_ribs(
                [i_n_coords, i_c_coords], [j_n_coords, j_c_coords]
            )
            taylor_scores["chothia_omega_angles"].append(omega_angle)

            # Getting letter packing descriptors:
            taylor_letter = get_taylor_letter_from_numeric_descriptor_and_omega(
                numeric_descriptor, omega_angle
            )
            taylor_scores["taylor_letter_packing_descriptors"].append(taylor_letter)

            # getting parallelism: returns
            parallelism = determine_vector_parallelism(angle_between_rib_vectors)

            # Getting distance_score. Favour close packing
            distance_score = get_distance_score(
                i_n_coords,
                i_c_coords,
                j_n_coords,
                j_c_coords,
                parallelism,
                deltahedron.edge_length,
            )
            taylor_scores["distance_scores"].append(distance_score)

            dihedral_angle = get_dihedral_angle(
                i_n_coords, i_c_coords, j_n_coords, j_c_coords
            )
            taylor_scores["dihedral_angles"].append(dihedral_angle)

            # Punishing paralel and favouring antiparalel
            # TODO: check if dihedral angle is any better to use instead of angle_between_rib_vectors
            orientation_score = get_orientation_score(angle_between_rib_vectors)
            taylor_scores["orientation_scores"].append(orientation_score)

            # Checking sequence proximity (0 or 1)
            sequence_proximity = get_sequence_proximity(i_helix_index, j_helix_index)

            taylor_scores["sequence_proximity"].append(sequence_proximity)

            # Getting get_combined_helix_pair_score
            combined_helix_pair_score = get_combined_helix_pair_score(
                orientation_score, sequence_proximity, distance_score
            )
            taylor_scores["combined_helix_pair_scores"].append(
                combined_helix_pair_score
            )

    # Sum up to overall_fold_score
    taylor_scores["overall_fold_score"] = get_taylor_overall_fold_score(
        taylor_scores["combined_helix_pair_scores"]
    )

    return taylor_scores


def test_get_tadas_scores_for_permutation():
    # TODO complete the test once confident it is working
    ribs = [[0, 1], [2, 3], [4, 5]]
    taylor_scores = get_tadas_scores_for_permutation(ribs)
    # assert ...


def get_taylor_letter_from_numeric_descriptor_and_omega(
    numeric_descriptor, omega_angle
):
    full_Taylor_descriptor_dict = {
        "1112": "a",
        "2111": "b",
        "1211": "c",
        "1121": "d",
        "1122": "e",
        "1221": "f",
        "2211": "g",
        "2121": "h",
        "1212": "i",
        "2112": "j",
    }
    try:
        letter = full_Taylor_descriptor_dict[numeric_descriptor]
    except KeyError:
        letter = "x"
    if omega_angle < 0:
        # murzin: N, chris n, Taylor: Z in general or alphabet uppercase for specific descriptors
        letter = letter.upper()
    elif omega_angle > 0:
        # murzin: И, chris i, Taylor: S in general or alphabet lowecase for specific descriptors.
        letter = letter.lower()
    else:
        # print(f"omega_angle is 0. Two helices are perfectly paralel")
        # unsure what to do in this case - it is neither right or left handed.
        letter = letter.upper()
    return letter


def get_taylor_numeric_descriptor(
    i_n_index, i_c_index, j_n_index, j_c_index, deltahedron
):
    nn = find_shortest_path(i_n_index, j_n_index, deltahedron)
    nc = find_shortest_path(i_n_index, j_c_index, deltahedron)
    cn = find_shortest_path(i_c_index, j_n_index, deltahedron)
    cc = find_shortest_path(i_c_index, j_c_index, deltahedron)

    assert all(
        [nn >= 0, nc >= 0, cn >= 0, cc >= 0]
    )  # find_shortest_path can output -1 if does not find a path
    assert all(
        [nn <= 3, nc <= 3, cn <= 3, cc <= 3]
    )  # max distance in any deltahedra is 3

    numeric_descriptor = f"{nn}{nc}{cn}{cc}"

    return numeric_descriptor


def test_get_taylor_numeric_descriptor(rib_len):
    # TODO visually confirm these tests and expand
    rib_num = 6
    deltahedron = Deltahedron.choose_deltahedron_by_rib_number(rib_num, rib_len)
    assert (
        get_taylor_numeric_descriptor(
            i_n_index=0, i_c_index=5, j_n_index=11, j_c_index=8, deltahedron=deltahedron
        )
        == "3223"
    )

    rib_num = 5
    deltahedron = Deltahedron.choose_deltahedron_by_rib_number(rib_num, rib_len)
    assert (
        get_taylor_numeric_descriptor(
            i_n_index=0, i_c_index=3, j_n_index=7, j_c_index=9, deltahedron=deltahedron
        )
        == "2312"
    )


def find_shortest_path(start, end, deltahedron: Deltahedron):
    connections = deltahedron.connections
    visited = [False] * len(deltahedron.vertices)
    # To store the number of steps from start to each vertex
    steps = [0] * len(deltahedron.vertices)
    queue = [start]
    visited[start] = True

    while queue:
        vertex = queue.pop(0)  # Get a vertex from the front of the queue
        if vertex == end:
            return steps[vertex]  # Return the number of steps from start to end

        for neighbor in connections[vertex]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                steps[neighbor] = (
                    steps[vertex] + 1
                )  # Increment the step count for the neighbor

    return -1  # Return -1 if there is no path from start to end


def test_find_shortest_path():
    # Visually confirmed with M&F 1988 paper.
    rib_len = 11
    rib_num = 6
    deltahedron = Deltahedron.choose_deltahedron_by_rib_number(rib_num, rib_len)
    assert find_shortest_path(start=0, end=11, deltahedron=deltahedron) == 3

    rib_num = 5
    deltahedron = Deltahedron.choose_deltahedron_by_rib_number(rib_num, rib_len)
    assert find_shortest_path(start=0, end=5, deltahedron=deltahedron) == 2

    rib_num = 4
    deltahedron = Deltahedron.choose_deltahedron_by_rib_number(rib_num, rib_len)
    assert find_shortest_path(start=0, end=5, deltahedron=deltahedron) == 1

    rib_num = 3
    deltahedron = Deltahedron.choose_deltahedron_by_rib_number(rib_num, rib_len)
    assert find_shortest_path(start=0, end=5, deltahedron=deltahedron) == 2


def rotate_about_axis(point, axis, angle):
    """
    Rotate a point counterclockwise by a given angle around a given axis.

    Parameters:
    - point: np.array, point to be rotated
    - axis: np.array, rotation axis
    - angle: float, rotation angle

    Returns:
    - np.array, rotated point
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    return np.dot(rotation_matrix, point)


def get_sequence_proximity(i_helix_index, j_helix_index):
    if j_helix_index - i_helix_index == 1:
        return 1
    else:
        return 0


def get_chothia_omega_angle_between_two_ribs(rib1, rib2):
    # validated with b3iii and b3nnn that must result in all negative and all positive angles respectively (60 degrees all)
    # from chothia et al:
    # We use Q to describe the relative orientation of two pieces of secondary structure in contact.
    # Q is defined as the angle between the strands of the pleated sheet and/or helix axes when projected onto their plane of contact.
    # We ignore the direction of individual α-helices and strands so 0 is defined between -90° and +90° rather than -180° and +180°.
    # The angle is negative (0° to -90°) if the near helix or strand is rotated in a clockwise direction relative to the far helix or strand.
    # If this rotation is anticlockwise, the angle is positive (0° to +90°).

    rib1, rib2 = np.array(rib1), np.array(rib2)

    # Align plane of contact to x-y plane
    rib1_vector = rib1[1] - rib1[0]
    rib2_vector = rib2[1] - rib2[0]
    normal_to_plane = np.cross(rib1_vector, rib2_vector)

    if np.all(
        normal_to_plane == 0
    ):  # When rib1_vector and rib2_vector are perfectly antiparallel or parallel, return chothia anlge = 0
        return 0

    angle_to_xy_plane = np.arccos(
        np.dot(normal_to_plane, [0, 0, 1]) / np.linalg.norm(normal_to_plane)
    )

    rotation_axis = np.cross(normal_to_plane, [0, 0, 1])

    # Only perform the rotation if rotation_axis is not [0, 0, 0] to avoid NaN values
    if not np.all(rotation_axis == 0):
        rib1 = np.array(
            [
                rotate_about_axis(point, rotation_axis, angle_to_xy_plane)
                for point in rib1
            ]
        )
        rib2 = np.array(
            [
                rotate_about_axis(point, rotation_axis, angle_to_xy_plane)
                for point in rib2
            ]
        )
    # Determine "near" and "far" rib based on z-value
    if np.mean(rib1[:, 2]) > np.mean(rib2[:, 2]):
        near_rib, far_rib = rib1, rib2
    else:
        near_rib, far_rib = rib2, rib1

    # Determine the angle and direction needed to rotate the "near" rib along the z-axis
    near_rib_vector = near_rib[1] - near_rib[0]
    far_rib_vector = far_rib[1] - far_rib[0]

    cosine_angle = np.dot(near_rib_vector, far_rib_vector) / (
        np.linalg.norm(near_rib_vector) * np.linalg.norm(far_rib_vector)
    )
    cosine_angle = np.clip(cosine_angle, -1, 1)  # clamp the value to [-1, 1]
    angle = np.arccos(cosine_angle)

    direction = np.sign(np.cross(near_rib_vector, far_rib_vector)[2])

    # Convert angle to degrees and adjust its range
    angle = np.degrees(angle) * direction * -1
    while angle > 90 or angle < -90:
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

    return round(angle, 1)


def test_get_chothia_omega_angle_between_two_ribs():
    # Parallel vectors pointing in the same direction (Expected: 0)
    v1 = [(0, 0, 0), (1, 0, 0)]
    v2 = [(0, 1, 0), (1, 1, 0)]
    assert abs(get_chothia_omega_angle_between_two_ribs(v1, v2)) == 0

    # Parallel vectors pointing in opposite direction (Expected: 0)
    v1 = [(0, 0, 0), (1, 0, 0)]
    v2 = [(0, 1, 0), (-1, 1, 0)]
    assert abs(get_chothia_omega_angle_between_two_ribs(v1, v2)) == 0

    # Perpendicular vectors (Expected: 90 or -90)
    v1 = [(0, 0, 0), (1, 0, 0)]
    v2 = [(0, 1, 0), (0, 1, 1)]
    assert abs(get_chothia_omega_angle_between_two_ribs(v1, v2)) == 90

    # Vectors at 45 degrees to each other
    v1 = [(0, 0, 0), (1, 0, 0)]
    v2 = [(0, 1, 0), (1, 1, 1)]
    assert get_chothia_omega_angle_between_two_ribs(v1, v2) == -45

    v1 = [(0, 1, 0), (1, 1, 0)]
    v2 = [(0, 0, 0), (1, 0, 1)]
    assert get_chothia_omega_angle_between_two_ribs(v1, v2) == 45


def determine_vector_parallelism(degree_angle_between_vectors):
    assert 0 <= degree_angle_between_vectors <= 180
    if degree_angle_between_vectors < 90:
        return "parallel"
    elif degree_angle_between_vectors > 90:
        return "antiparallel"
    elif degree_angle_between_vectors == 90:
        return "perpendicular"


def test_determine_vector_parallelism():
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])  # v2 is just a scaled version of v1, so they are parallel
    assert determine_vector_parallelism(v1, v2) == "parallel"

    v1 = np.array([1, 2, 3])
    v2 = np.array(
        [-1, -2, -3]
    )  # v2 is just a negative scaled version of v1, so they are antiparallel
    assert determine_vector_parallelism(v1, v2) == "antiparallel"

    v1 = np.array([1, 0, 0])  # This is along the x-axis
    v2 = np.array([0, 1, 0])  # This is along the y-axis, so v1 and v2 are perpendicular
    assert determine_vector_parallelism(v1, v2) == "perpendicular"


def get_distance_score(
    i_n_coords, i_c_coords, j_n_coords, j_c_coords, parallelism, rib_len
):
    # this is similar to distance score in Taylor et al paper, however, it is a normalized, inverted version of the score.
    # Supposed to favour close packing of two helices as well as minimal loop length.
    # distance_score value range are [rib_length, +infinity)
    # thus distance_score_inverted_normalized value range is [1, 0)
    # Minimal distance between longitudinal interaction (parallel/antiparalel) will be 1. Suboptimal distances will result in >1

    # TODO could plot over multiple rib lengths deltaprots to see how the score behaves
    if parallelism == "parallel":
        # nn + cc - d
        distance_score = (
            euclidian_distance(i_n_coords, j_n_coords)
            + euclidian_distance(i_c_coords, j_c_coords)
            - rib_len
        )
    elif parallelism == "antiparallel":
        # nc + cn - d
        distance_score = (
            euclidian_distance(i_n_coords, j_c_coords)
            + euclidian_distance(i_c_coords, j_n_coords)
            - rib_len
        )
    elif parallelism == "perpendicular":
        parallel_distance_score = (
            euclidian_distance(i_n_coords, j_n_coords)
            + euclidian_distance(i_c_coords, j_c_coords)
            - rib_len
        )
        antiparallel_distance_score = (
            euclidian_distance(i_n_coords, j_c_coords)
            + euclidian_distance(i_c_coords, j_n_coords)
            - rib_len
        )
        distance_score = min(parallel_distance_score, antiparallel_distance_score)
    else:
        print(f"unexpected parallelism: {parallelism}")

    distance_score_inverted_normalized = rib_len / distance_score
    return round(distance_score_inverted_normalized, 3)


def get_dihedral_angle(i_n_coords, i_c_coords, j_n_coords, j_c_coords):
    # Note the order of the coordinates:
    # The points a, b and c form a plane in R^3. The points b, c and d form a plane in R^3. The dihedral angle is the angle between these two planes.
    # a is i_N point, b is i midpoint, c is j midppoint, d is j_N point
    i_midpoint = (i_n_coords + i_c_coords) / 2
    j_midpoint = (j_n_coords + j_c_coords) / 2
    dihedral_angle = dihedral(i_n_coords, i_midpoint, j_midpoint, j_n_coords)

    return round(dihedral_angle, 1)


def get_orientation_score(angle):
    # the whole goal of this score is to punish paralel helices and favour antiparalel helices.
    # make output = 1 on perpendicular case
    # output = 0.5 on paralel case
    # output = 1.5 on antiparalel case
    # values always above 0
    # Suitable function for this: 1 - cos(x)/2, where x is dihedral angle
    # This function will have range of values: [0.5, 1.5]

    return round(1 - np.cos(math.radians(angle)) / 2, 3)


def test_get_orientation_score():
    # Tolerance for floating point comparison
    tolerance = 1e-6

    # 1. Test for parallel vectors
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([1, 1, 0])
    p4 = np.array([0, 1, 0])
    score = get_orientation_score(p1, p2, p3, p4)
    assert np.abs(score - 0.5) < tolerance

    # 2. Test for antiparallel vectors
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([-1, -1, 0])
    p4 = np.array([0, -1, 0])
    score = get_orientation_score(p1, p2, p3, p4)
    assert np.abs(score - 1.5) < tolerance

    # 3. Test for perpendicular vectors
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([1, 1, 0])
    p4 = np.array([1, 1, 1])
    score = get_orientation_score(p1, p2, p3, p4)
    assert np.abs(score - 1) < tolerance


def get_combined_helix_pair_score(
    orientation_score, sequence_proximity, distance_score
):
    # if orientation_score [1.5, 0.5] and distance_score [1,0), the combined_helix_pair_score has range of values [1.5,0)
    # punish via orientation_score only if the helices are in sequence (sequence_proximity=1)

    if sequence_proximity == 1:
        # helices are consecutive
        combined_helix_pair_score = orientation_score * distance_score
    elif sequence_proximity == 0:
        combined_helix_pair_score = 1 * distance_score

    return round(combined_helix_pair_score, 2)


def get_taylor_overall_fold_score(combined_orientation_scores):
    # make sure the passed combined_orientation_scores already lacks the i>j combinations avoiding redundancy as well as i!=j avoiding self comparison of helices
    overall_fold_score = sum(combined_orientation_scores)
    return round(overall_fold_score, 2)
