import dataclasses
import numpy
import json
import os

# import the relevant secondary structure
from ampal.assembly import Assembly
from isambard.specifications.helix import Helix
from ampal.geometry import dihedral
import isambard.modelling as modelling
from typing import Union, List, Tuple


def gen_octahedron(el):
    """Generates vertices of an octahedron with a defined edge length.

    Parameters
    ----------
    el : float
        The edge length of the polyhedron in arbitrary units.

    Returns
    -------
    vertices : [triple]
        List containing coordinates of the vertices of the polyhedron.
    """
    xy = numpy.sin(numpy.pi / 4) * el
    vertices = [
        (0, 0, xy),  # a
        (0, xy, 0),  # b
        (xy, 0, 0),  # c
        (0, -xy, 0),  # d
        (-xy, 0, 0),  # e
        (0, 0, -xy),  # f
    ]
    return vertices


def gen_snub_disphenoid(el):
    """Generates vertices of a snub-nosed disphenoid with a defined edge length.

    Parameters
    ----------
    el : float
        The edge length of the polyhedron in arbitrary units.

    Returns
    -------
    vertices : [triple]
        List containing coordinates of the vertices of the polyhedron.
    """
    # -z1 to move gh vector off x axis
    x2 = 0.644584 * el
    z1 = 0.578369 * el
    z2 = 0.989492 * el
    z3 = 1.56786 * el

    vertices = [
        (0, el / 2, z3 - z1),  # a
        (0, -el / 2, z3 - z1),  # b
        (0, -x2, z1 - z1),  # c
        (-x2, 0, z2 - z1),  # d
        (0, x2, z1 - z1),  # e
        (x2, 0, z2 - z1),  # f
        (el / 2, 0, 0 - z1),  # g
        (-el / 2, 0, 0 - z1),  # h
    ]
    return vertices


def gen_gyro_square_bipyramid(el):
    """Generates vertices of a gyroelongated bipyramid with a defined edge length.

    Parameters
    ----------
    el : float
        The edge length of the polyhedron in arbitrary units.

    Returns
    -------
    vertices : [(float, float, float)]
        List containing coordinates of the vertices of the polyhedron.
    """
    rl = (0.5 * el) / numpy.sin(numpy.pi / 4)
    zs = numpy.sin(numpy.pi / 3) * el
    pl = rl - (el / 2)
    z1 = numpy.sqrt((zs**2) - (pl**2)) / 2
    rxy = el / 2
    theta = numpy.arccos(rl / el)
    z2 = numpy.sin(theta) * el
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


def gen_icosahedron(el):
    """Generates vertices of an icosahedron with a defined edge length.

    Parameters
    ----------
    el : float
        The edge length of the polyhedron in arbitrary units.

    Returns
    -------
    vertices : [(float, float, float)]
        List containing coordinates of the vertices of the polyhedron.
    """
    rl = (el / 2) / numpy.sin(numpy.pi / 5)
    x2 = numpy.cos(numpy.pi / 2 - (2 * numpy.pi / 5)) * rl
    y2 = numpy.sin(numpy.pi / 2 - (2 * numpy.pi / 5)) * rl
    x3 = numpy.sin(numpy.pi - 2 * (2 * numpy.pi / 5)) * rl
    y3 = numpy.cos(numpy.pi - 2 * (2 * numpy.pi / 5)) * rl
    x4 = numpy.sin(numpy.pi / 5) * rl
    y4 = numpy.cos(numpy.pi / 5) * rl
    x5 = numpy.cos((3 * numpy.pi / 5) - (numpy.pi / 2)) * rl
    y5 = numpy.sin((3 * numpy.pi / 5) - (numpy.pi / 2)) * rl
    zs = numpy.sqrt(el**2 - (el / 2) ** 2)
    z1 = numpy.sqrt(zs**2 - (rl - y4) ** 2) / 2
    z2 = numpy.sqrt(el**2 - rl**2)
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


def get_octahedron_connections():
    return [
        [1, 2, 3, 4],
        [0, 2, 4, 5],
        [0, 1, 3, 5],
        [0, 2, 4, 5],
        [0, 1, 3, 5],
        [1, 2, 3, 4],
    ]


def get_snub_disphenoid_connections():
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


def get_gyro_square_bipyramid_connections():
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


def get_icosahedron_connections():
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


@dataclasses.dataclass
class HelixConformation:
    """Contains parameters for an individual helix of a `DeltaProt`"""

    rib_vertices: Tuple[int, int]
    helix_axis_rotation: float
    num_amino_acids: int


"""
my_dp = DeltaProt(
    helix_conformations=[
        HelixConformation((0, 1), 330.0, 11),
        HelixConformation((2, 3), 330.0, 11),
        HelixConformation((4, 5), 330.0, 11),
    ],
    rib_len=11,
    centre_helices=True,
    centred_ca=1,
)
my_dp.determine_orientation_code() # might return the ways that its wrong

my_nice_dp = DeltaProt.from_conformation_string("b3iii")
"""


class DeltaProt(Assembly):
    """Generates a deltahedral protein with specified rib orientation and length.

    Notes
    -----
    If centre_helices is True then the helices will be translated along the ribs so the centre of the helices is
    equal to the centre of the rib. They will also be rotated so that the middle most residue (rounded-up) will be
    rotated to face the centre of the assembly.

    All distances in angstroms.

    Parameters
    ----------
    conformation : string
        String describing the deltaprot form to be modelled. See the keys of the rib_orient dictionary for details of
        the various forms available.
    aa : int
        Number of amino acids in the individual helices of the assembly. Keyword argument, default value = 8.
    centre_helices : bool
        If centre_helices is True then the helices will be translated along the ribs so the centre of the helices is
        equal to the centre of the rib. They will also be rotated so that the middle most residue (rounded-up) will be
        rotated to face the centre of the assembly. Keyword argument, default value = True.
    centred_ca : int
        Residue on which the helices are centred. Keyword argument, default value = 1.

    Attributes
    ----------
    ax_trans_v : [float]

    assembly : [[(float, float, float)]]
        List containing a list for each chain in the assembly, each of which contain the coordinates of the mainchain/cb
        atoms.
    """

    rib_orientations = get_rib_orientations()
    orientation_codes = get_orientation_codes(with_dots=False)
    orientation_codes_w_dots = get_orientation_codes(with_dots=True)

    default_rib_len = 11
    default_aa = 10
    # problem: will set this value for all of instances of a class
    # TODO: use clasmethod instead. Or staticmethod?

    choose_delta = {
        3: gen_octahedron,
        4: gen_snub_disphenoid,
        5: gen_gyro_square_bipyramid,
        6: gen_icosahedron,
    }

    get_connections = {
        3: get_octahedron_connections,
        4: get_snub_disphenoid_connections,
        5: get_gyro_square_bipyramid_connections,
        6: get_icosahedron_connections,
    }

    def __init__(
        self,
        helix_conformations: List[HelixConformation],
        rib_len: float = None,
        centre_helices: bool = True,
        centred_ca: int = 1,
    ):

        super(DeltaProt, self).__init__()  # keep Assembly init and append this init

        self.helix_conformations = helix_conformations
        self.rib_len = rib_len if rib_len is not None else self.default_rib_len
        if centre_helices:
            self.ax_trans_adjust = [
                (self.rib_len / 2.0)
                - (((self.helix_conformations[i].num_amino_acids - 1) * 1.52) / 2.0)
                for i in range(len(self.helix_conformations))
            ]
        else:
            self.ax_trans_adjust = [0] * len(self.helix_conformations)
        self.centred_ca = centred_ca - 1

        self.orientation_code = self.determine_orientation_code()
        if self.orientation_code is not None:
            print(
                f"Created an assembly with Murzin & Finkelstein orientation code '{self.orientation_code}'"
            )
        else:
            print(f"Created an assembly with unknown orientation")

        self.build()

    def helices_edges(self):
        dv = self.deltahedron_vertices()
        helices_edges = []
        for helix_conformation in self.helix_conformations:
            v1, v2 = helix_conformation.rib_vertices
            helices_edges.append((dv[v1], dv[v2]))
        return helices_edges

    def loops_edges(self):
        dv = self.deltahedron_vertices()
        rib_vertices = self.ribs
        loops_edges = []
        for i in range(len(rib_vertices) - 1):
            v1, v2 = rib_vertices[i]
            v1_next, v2_next = rib_vertices[i + 1]
            loops_edges.append((dv[v2], dv[v1_next]))
        return loops_edges

    def deltahedron_vertices(self):
        # Number and length of ribs determine the vertices of the deltaprot shape
        return self.choose_delta[len(self.helix_conformations)](self.rib_len)

    @property
    def centre(self):
        dv = self.deltahedron_vertices()
        centre = sum([numpy.array(x) for x in dv]) / len(dv)
        return centre

    def build(self):
        """Uses input parameters to build model in selected OldDeltaProt topology.

        Rerunning build overwrites previous assembly."""
        polymers = []
        for i, (start, end) in enumerate(self.helices_edges()):
            start = numpy.array(start)
            end = numpy.array(end)
            helix = Helix(aa=self.helix_conformations[i].num_amino_acids)
            helix.move_to(start=start, end=end)
            helix.translate(self.ax_trans_adjust[i] * helix.axis.unit_tangent)
            ax_rot = dihedral(
                self.centre, end, start, helix[self.centred_ca]["CA"]._vector
            )
            helix.rotate(
                angle=ax_rot, axis=helix.axis.unit_tangent, point=helix.axis.midpoint
            )  # initial rotation of helix aligning a specific residue to face the bundle centre (might be redundant)

            helix.rotate(
                angle=self.helix_conformations[i].helix_axis_rotation,
                axis=helix.axis.unit_tangent,
                point=helix.axis.midpoint,
            )
            polymers.append(helix)

        # TODO make Assembly method reset_ampal_parents. (ampal_children?) May need quivalent at polymer level.
        self._molecules = polymers[:]
        for polymer in self._molecules:
            polymer.ampal_parent = self
            for monomer in polymer._monomers:
                monomer.ampal_parent = polymer
        self.relabel_polymers()  # relabel to give each helix a chain label
        self.relabel_atoms()

        return

    def update_with_new_model(self, new_model):
        for old, new in zip(self._molecules, new_model._molecules):
            old._monomers = new._monomers

    def determine_orientation_code(self):
        orientation_codes_sorted_ribs = {
            "b3iii": [(0, 1), (2, 3), (4, 5)],
            "b3nnn": [(0, 2), (1, 4), (3, 5)],
            "b4iiiix": [(0, 5), (1, 3), (2, 7), (4, 6)],
            "b4iiiiy": [(0, 1), (2, 5), (3, 4), (6, 7)],
            "b4iiin": [(0, 4), (1, 5), (2, 6), (3, 7)],
            "b4inin": [(0, 4), (1, 2), (3, 7), (5, 6)],
            "b4innn": [(0, 5), (1, 2), (3, 7), (4, 6)],
            "b4nnnnx": [(0, 3), (1, 5), (2, 6), (4, 7)],
            "b4nnnny": [(0, 1), (2, 3), (4, 5), (6, 7)],
            "b5iiiin": [(0, 4), (1, 2), (3, 8), (5, 6), (7, 9)],
            "b5iinin": [(0, 4), (1, 5), (2, 6), (3, 8), (7, 9)],
            "b5ininn": [(0, 1), (2, 6), (3, 8), (4, 5), (7, 9)],
            "b5innnn": [(0, 1), (2, 6), (3, 4), (5, 8), (7, 9)],
            "b6iiniin": [(0, 2), (1, 7), (3, 8), (4, 9), (5, 10), (6, 11)],
            "b6ininin": [(0, 3), (1, 7), (2, 8), (4, 9), (5, 10), (6, 11)],
            "b6inninn": [(0, 3), (1, 2), (4, 9), (5, 10), (6, 11), (7, 8)],
            "h4i.n": [(0, 5), (1, 3), (2, 6), (4, 7)],
            "h5i.i": [(0, 1), (2, 3), (4, 8), (5, 6), (7, 9)],
            "h5n.n": [(0, 4), (1, 6), (2, 3), (5, 8), (7, 9)],
            "h6i.i.i": [(0, 4), (1, 2), (3, 9), (5, 6), (7, 8), (10, 11)],
            "h6n.n.n": [(0, 1), (2, 7), (3, 4), (5, 10), (6, 11), (8, 9)],
            "l4iin": [(0, 1), (2, 6), (3, 7), (4, 5)],
            "l4inn": [(0, 1), (2, 5), (3, 7), (4, 6)],
            "l5iiin": [(0, 2), (1, 6), (3, 4), (5, 8), (7, 9)],
            "l5inni": [(0, 2), (1, 6), (3, 8), (4, 5), (7, 9)],
            "l5innn": [(0, 3), (1, 4), (2, 6), (5, 8), (7, 9)],
            "l5niin": [(0, 3), (1, 5), (2, 6), (4, 8), (7, 9)],
            "l6innni": [(0, 1), (2, 3), (4, 9), (5, 10), (6, 11), (7, 8)],
            "l6niiin": [(0, 1), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11)],
            "s6": [(0, 5), (1, 7), (2, 3), (4, 9), (6, 10), (8, 11)],
        }

        # Match helix_conformation.rib_vertices against sorted orientation codes
        rib_vertices = [i.rib_vertices for i in self.helix_conformations]

        sorted_rib_vertices = sorted([tuple(sorted(pair)) for pair in rib_vertices])
        determined_orientation_code = None
        for code, orientation_sorted_ribs in orientation_codes_sorted_ribs.items():
            if sorted_rib_vertices == orientation_sorted_ribs:
                determined_orientation_code = code  # Return the matching code
        return determined_orientation_code


__author__ = "Christopher W. Wood"
__status__ = "Development"
