import dataclasses
import numpy
import warnings

# import the relevant secondary structure
from ampal.assembly import Assembly
from isambard.specifications.helix import Helix
from ampal.geometry import dihedral
from typing import List, Tuple
from isambard.specifications.deltaprot_helper import (
    get_tadas_scores_for_permutation,
    choose_delta,
    get_rib_orientations,
    get_orientation_codes,
)


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
    # TODO: Remove these completely (I could substitute these in project config  instead)

    def __init__(
        self,
        helix_conformations: List[HelixConformation],
        rib_len: float = None,
        centre_helices: bool = True,
        centred_ca: int = 1,
    ):

        super(DeltaProt, self).__init__()  # keep Assembly init and append this init

        assert len(helix_conformations) in [
            3,
            4,
            5,
            6,
        ], "number of helix_conformations must be between 3-6. Deltahedron size is prescribed based on this number. Assembling incomplete folds is not yet implemented."
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

    def get_tadas_assembly_score(self):

        if self.orientation_code is None:
            warnings.warn(
                "Tadas score is only tested for complete Murzin & Finkelstein orientations",
                UserWarning,
            )

        ribs_sequence = [
            helix_conformation.rib_vertices
            for helix_conformation in self.helix_conformations
        ]

        # A score for rib arrangement. Order of ribs matter. Direction of ribs matter. Helix axis rotation has no effect.
        # Inspired by Taylor et al.scoring.
        # Tadas score can be used to rank permutations of single chain M&F orientations.
        # Tadas score is not affected by helix rotation as looks to helices simply as ribs in space connected in sequence.
        return get_tadas_scores_for_permutation(ribs_sequence, self.rib_len)

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
        return choose_delta[len(self.helix_conformations)](self.rib_len)

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


__author__ = "Tadas Kluonis"
__status__ = "Development"


############
