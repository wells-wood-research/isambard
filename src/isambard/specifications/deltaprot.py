import numpy
import json
import os

# import the relevant secondary structure
from ampal.assembly import Assembly
from isambard.specifications.helix import Helix
from ampal.geometry import dihedral
import isambard.modelling as modelling
from typing import Union, List, Tuple

# TODO Remove rosetta mode, move average_2_points to geometry


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
    # problem will set this value for all of instances of a class
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
        conformation: str,
        rib_len: float = None,
        aa: Union[List, Tuple, int] = None,
        centre_helices: bool = True,
        centred_ca: int = 1,
        build_from_aa: str = "A",
        ribs: List = None,
        angles: List = None,
    ):

        super(DeltaProt, self).__init__()  # keep Assembly init and append this init

        conformation = conformation.lower()
        if conformation not in self.rib_orientations.keys():
            raise ValueError("Invalid deltaprot conformation {}".format(conformation))
        else:
            self.conformation = conformation

        # Use provided ribs, angles, aa if they are not None, else use the default values
        self.rib_len = rib_len if rib_len is not None else self.default_rib_len
        self.ribs = (
            ribs
            if ribs is not None
            else self.rib_orientations[self.conformation]["ribs"]
        )
        self.angles = (
            angles
            if angles is not None
            else self.rib_orientations[self.conformation]["angles"]
        )
        if isinstance(aa, int):
            self.aa = [aa] * len(self.ribs)
        elif isinstance(aa, List) or isinstance(aa, Tuple):
            self.aa = aa
        elif aa == None:
            self.aa = [self.default_aa] * len(self.ribs)

        self.rib_num = int(self.conformation[1])
        # self.ap = [0] * self.rib_num
        self.centred_ca = centred_ca - 1
        if centre_helices:
            self.ax_trans_adjust = [
                (self.rib_len / 2.0) - (((self.aa[i] - 1) * 1.52) / 2.0)
                for i in range(len(self.ribs))
            ]
        else:
            self.ax_trans_adjust = [0] * len(self.ribs)

        self.build_from_aa = build_from_aa

        self.build()

    def helices_edges(self):
        dv = self.deltahedron_vertices()
        # vertices to choose specific to the conformation
        rib_vertices = self.ribs
        helices_edges = []
        for i, vertices in enumerate(rib_vertices):
            v1, v2 = vertices
            # flip if antiparallel
            # if self.ap[i]:
            #     edges.append((dv[v2], dv[v1]))
            # else:
            helices_edges.append(
                (dv[v1], dv[v2])
            )  # Tadas changes: ignore antiparalel flag
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
        return self.choose_delta[self.rib_num](self.rib_len)

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
            helix = Helix(aa=self.aa[i])
            helix.move_to(start=start, end=end)
            helix.translate(self.ax_trans_adjust[i] * helix.axis.unit_tangent)
            ax_rot = dihedral(
                self.centre, end, start, helix[self.centred_ca]["CA"]._vector
            )
            helix.rotate(
                angle=ax_rot, axis=helix.axis.unit_tangent, point=helix.axis.midpoint
            )
            helix.rotate(
                angle=self.angles[i],
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
        self.relabel_polymers()  # relabel to give each a chain label
        self.relabel_atoms()

        if self.build_from_aa != "G":
            # print("before",self[0].axis)
            # polypeptide_count = len([i for i in self])
            # model_sequences = polypeptide_count * [self.build_from_aa * self.aa]
            model_sequences = [
                self.build_from_aa * res_num for res_num in self.aa
            ]  # changes when introduced aa as a list
            # model_sequences = [self.build_from_aa * len(list(self.get_monomers()))]
            all_aa_model = modelling.pack_side_chains_scwrl(self, model_sequences)
            self.update_with_new_model(all_aa_model)
            # print("after update",self[0].axis)
        return

    def update_with_new_model(self, new_model):
        for old, new in zip(self._molecules, new_model._molecules):
            old._monomers = new._monomers


__author__ = "Christopher W. Wood"
__status__ = "Development"
