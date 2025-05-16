import dataclasses
import numpy
import warnings

# import the relevant secondary structure
from ampal.assembly import Assembly
from isambard.specifications.helix import Helix
from ampal.geometry import dihedral
from typing import List, Tuple
from isambard.specifications.deltaprot_helper import (
    get_path_scores_for_permutation,
    # choose_delta,
    get_rib_orientations,
    get_orientation_codes,
    Deltahedron,
    find_shortest_path,
    custom_formatwarning,
    get_retained_symmetry_axes,
    get_hydrophobic_count,
    get_CA_CB_phantom_vectors,
    find_middle_degree_of_largest_cluster_of_max_values,
    get_MF_orientation_code_from_rib_vertices,
)

warnings.formatwarning = custom_formatwarning


@dataclasses.dataclass
class HelixConformation:
    """
    Contains parameters for an individual helix of a `DeltaProt`
    """

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
    edge_length=11,
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
    helix_conformations : list
        List of HelixConformation objects. Order of the list determines the N-C order of the helices in the assembly.
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

    default_edge_length = 11
    default_aa = 10
    # TODO: Remove these completely (I could substitute these in project config  instead)

    def __init__(
        self,
        helix_conformations: List[HelixConformation],
        deltahedron_name: str = None,
        edge_length: float = None,
        centre_helices: bool = True,
        centred_ca: int = 1,
        chain_label: str = None,
    ):

        super(DeltaProt, self).__init__()  # keep Assembly init and append this init

        self.helix_conformations = helix_conformations
        edge_length = self.default_edge_length if edge_length is None else edge_length

        if deltahedron_name is None:
            assert len(self.helix_conformations) in [
                3,
                4,
                5,
                6,
            ], "Number of HelixConformations passed must be 3-6 if deltahedron_name is not specified."
            self.deltahedron = Deltahedron.choose_deltahedron_by_rib_number(
                len(self.helix_conformations), edge_length
            )
        else:
            self.deltahedron = Deltahedron.choose_deltahedron_by_name(
                deltahedron_name, edge_length
            )

        if centre_helices:
            self.ax_trans_adjust = [
                (self.deltahedron.edge_length / 2.0)
                - (((self.helix_conformations[i].num_amino_acids - 1) * 1.52) / 2.0)
                for i in range(len(self.helix_conformations))
            ]
        else:
            self.ax_trans_adjust = [0] * len(self.helix_conformations)

        self.centred_ca = centred_ca - 1

        self.orientation_code = self.get_MF_orientation_code()

        self.check_assembly_quality()

        self.build()

        if chain_label is not None:
            residue_id = 1
            for chain in self:
                chain.id = chain_label
                for residue in chain:
                    residue.id = residue_id
                    residue_id += 1

    def get_chiral_rib_symmetry(self):
        # Ignores miror, improper rotations, inversions as they dont make sense for a chiral helix.
        # Only looks at cyclic rotational symmetries
        # Assumes that assembly symmetry will be a subset of deltahedron symmetry as the helices touch every vertex of deltahedron
        rib_vertices = [
            helix_conf.rib_vertices for helix_conf in self.helix_conformations
        ]
        return get_retained_symmetry_axes(
            rib_vertices,
            self.deltahedron.symmetry_axes,
            # self.helices_edges(),
            self.deltahedron.vertices,
        )

    def check_assembly_quality(self):

        # Report what orientation code was built if any.
        if self.orientation_code is not None:
            print(
                f"Created an assembly with Murzin & Finkelstein orientation code '{self.orientation_code}'"
            )
        else:
            print(f"Created an assembly with unknown orientation")

        # Warn if any of the ribs are not on the surface of deltahedron
        if not all(
            find_shortest_path(
                conf.rib_vertices[0], conf.rib_vertices[1], self.deltahedron
            )
            == 1
            for conf in self.helix_conformations
        ):
            warnings.warn(
                "There are ribs crossing the core of deltahedron. All ribs should lie on the surface.",
                UserWarning,
            )

        vertice_list = [
            vertex for conf in self.helix_conformations for vertex in conf.rib_vertices
        ]
        if len(vertice_list) != len(set(vertice_list)):
            warnings.warn(
                "There are overlapping helix enpoints (rib vertices).",
                UserWarning,
            )

    def get_path_assembly_score(self):

        if self.orientation_code is None:
            warnings.warn(
                "path score is only tested for complete Murzin & Finkelstein orientations",
                UserWarning,
            )

        ribs_sequence = [
            helix_conformation.rib_vertices
            for helix_conformation in self.helix_conformations
        ]

        return get_path_scores_for_permutation(ribs_sequence, self.deltahedron)

    def helices_edges(self):
        helices_edges = []
        for helix_conformation in self.helix_conformations:
            v1, v2 = helix_conformation.rib_vertices
            alowed_indices = range(len(self.deltahedron.vertices))
            assert (
                v1 in alowed_indices and v2 in alowed_indices
            ), f"rib vertices {v1} and {v2} in helix conformations must be in range of {self.deltahedron.name} vertices indices {alowed_indices[0]}-{alowed_indices[-1]}"
            try:
                helices_edges.append(
                    (self.deltahedron.vertices[v1], self.deltahedron.vertices[v2])
                )
            except IndexError as e:
                e
        return helices_edges

    def loops_edges(self):
        loops_edges = []
        for i in range(len(self.helix_conformations) - 1):
            v1, v2 = self.helix_conformations[i].rib_vertices
            v1_next, v2_next = self.helix_conformations[i + 1].rib_vertices
            # for i in range(len(rib_vertices) - 1):
            #     v1, v2 = rib_vertices[i]
            #     v1_next, v2_next = rib_vertices[i + 1]
            loops_edges.append(
                (self.deltahedron.vertices[v2], self.deltahedron.vertices[v1_next])
            )
        return loops_edges

    @property
    def centre(self):
        centre = sum([numpy.array(x) for x in self.deltahedron.vertices]) / len(
            self.deltahedron.vertices
        )
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

    def get_MF_orientation_code(self):
        rib_vertices = [i.rib_vertices for i in self.helix_conformations]
        if len(rib_vertices) != self.deltahedron.rib_num:
            return None
        orientation_code = get_MF_orientation_code_from_rib_vertices(rib_vertices)
        return orientation_code

    def optimise_helix_rotations(self, degree_turn=5) -> None:
        """
        Optimises the helix rotations for the assembly and updates helix_conformations with the resulting angles.
        """
        current_hydrophobic_count = 0
        optimised_hydrophobic_count = 0
        for i, helix in enumerate(self):
            current_hydrophobic_count += get_hydrophobic_count(
                get_CA_CB_phantom_vectors(helix), self.deltahedron.vertices
            )

            helix_data = []
            degrees_rotated = 0
            assert 360 % degree_turn == 0, f" {degree_turn} must divide 360"
            while degrees_rotated < 360.0:
                helix.rotate(
                    angle=degree_turn,
                    axis=helix.axis.direction_vector,
                    point=helix.axis.midpoint,
                )
                degrees_rotated += degree_turn

                hydrophobic_count = get_hydrophobic_count(
                    get_CA_CB_phantom_vectors(helix), self.deltahedron.vertices
                )
                helix_data.append([degrees_rotated, hydrophobic_count])

            avg_optimal_angle, max_value = (
                find_middle_degree_of_largest_cluster_of_max_values(helix_data)
            )
            assert degrees_rotated == 360
            # since helix is now rotated 360, simply apply the optimal rotation to helix
            helix.rotate(
                angle=avg_optimal_angle,
                axis=helix.axis.direction_vector,
                point=helix.axis.midpoint,
            )

            optimised_hydrophobic_count += get_hydrophobic_count(
                get_CA_CB_phantom_vectors(helix), self.deltahedron.vertices
            )

            # Update the helix_conformations with the resulting angle
            self.helix_conformations[i].helix_axis_rotation = avg_optimal_angle

        print(
            f"Initial hydrophobic count: {current_hydrophobic_count}, optimised hydrophobic count: {optimised_hydrophobic_count}"
        )


# my_dp = DeltaProt(
#     [
#         HelixConformation((1, 2), 330, 10),
#         HelixConformation((3, 4), 330, 10),
#         HelixConformation((4, 11), 330, 10),
#     ],
#     deltahedron_name="icosahedron",
# )


__author__ = "Tadas Kluonis"
__status__ = "Development"
