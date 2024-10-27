from typing import List, Optional, Union, Dict, Tuple, Set
import logging
import numpy as np
import pydicom
import SimpleITK as sitk
import enum
from datetime import datetime

from pydicom.uid import SegmentationStorage
import abc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CodeSequence(pydicom.Sequence):
    """Helper class for constructing a DICOM CodeSequence."""

    def __init__(self, value: str, scheme_designator: str, meaning: str):
        """Creates a code sequence from mandatory arguments.

        Args:
            value: (0x0008, 0x0100) CodeValue
            scheme_designator: (0x0008, 0x0102) CodingSchemeDesignator
            meaning: (0x0008, 0x0104) CodeMeaning
        """
        super().__init__()
        ds = pydicom.Dataset()
        ds.CodeValue = value
        ds.CodingSchemeDesignator = scheme_designator
        ds.CodeMeaning = meaning
        self.append(ds)


class DimensionOrganizationSequence(pydicom.Sequence):
    def add_dimension(
        self,
        dimension_index_pointer: Union[str, pydicom.tag.Tag],
        functional_group_pointer: Optional[Union[str, pydicom.tag.Tag]] = None,
    ) -> None:
        ds = pydicom.Dataset()
        if len(self) > 0:
            ds.DimensionOrganizationUID = self[0].DimensionOrganizationUID
        else:
            ds.DimensionOrganizationUID = pydicom.uid.generate_uid()

        if isinstance(dimension_index_pointer, str):
            dimension_index_pointer = pydicom.tag.Tag(
                pydicom.datadict.tag_for_keyword(dimension_index_pointer)
            )
        ds.DimensionIndexPointer = dimension_index_pointer
        ds.DimensionDescriptionLabel = (
            pydicom.datadict.keyword_for_tag(dimension_index_pointer)
            or f"Unknown tag {dimension_index_pointer}"
        )

        if functional_group_pointer is not None:
            if isinstance(functional_group_pointer, str):
                functional_group_pointer = pydicom.tag.Tag(
                    pydicom.datadict.tag_for_keyword(functional_group_pointer)
                )
            ds.FunctionalGroupPointer = functional_group_pointer

        self.append(ds)


def dcm_to_sitk_orientation(iop: List[str]) -> np.ndarray:
    assert len(iop) == 6

    # Extract x-vector and y-vector
    x_dir = [float(x) for x in iop[:3]]
    y_dir = [float(x) for x in iop[3:]]

    # L2 normalize x-vector and y-vector
    x_dir /= np.linalg.norm(x_dir)  # type: ignore
    y_dir /= np.linalg.norm(y_dir)  # type: ignore

    # Compute perpendicular z-vector
    z_dir = np.cross(x_dir, y_dir)

    return np.stack([x_dir, y_dir, z_dir], axis=1)


def sitk_to_dcm_orientation(img: sitk.Image) -> List[str]:
    direction = img.GetDirection()
    assert len(direction) == 9
    direction = np.asarray(direction).reshape((3, 3))
    orientation = direction.T[:2]
    return [f"{x:e}" for x in orientation.ravel()]

def get_segment_map(dataset: pydicom.Dataset) -> Dict[int, pydicom.Dataset]:
    result: Dict[int, pydicom.Dataset] = {}
    last_number = 0
    for segment in dataset.SegmentSequence:
        if segment.SegmentNumber in result:
            raise ValueError(
                f"Segment {segment.SegmentNumber} was declared more than once."
            )

        if segment.SegmentNumber == 0:
            raise ValueError("Segment numbers must be start at 1.")

        if segment.SegmentNumber <= last_number:
            logger.warning(
                "Segment numbering should be monotonically increasing (last=%d, current=%d)",
                last_number,
                segment.SegmentNumber,
            )

        result[segment.SegmentNumber] = segment
        last_number = segment.SegmentNumber
    return result


def get_declared_image_spacing(dataset: pydicom.Dataset) -> Tuple[float, float, float]:
    sfg = dataset.SharedFunctionalGroupsSequence[0]
    if "PixelMeasuresSequence" not in sfg:
        raise ValueError("Pixel measures FG is missing!")

    pixel_measures = sfg.PixelMeasuresSequence[0]
    # DICOM defines (row spacing, column spacing) -> (y, x)
    y_spacing, x_spacing = pixel_measures.PixelSpacing
    if "SpacingBetweenSlices" in pixel_measures:
        z_spacing = pixel_measures.SpacingBetweenSlices
    else:
        z_spacing = pixel_measures.SliceThickness

    return float(x_spacing), float(y_spacing), float(z_spacing)


def get_image_direction(dataset: pydicom.Dataset) -> np.ndarray:
    sfg = dataset.SharedFunctionalGroupsSequence[0]
    if "PlaneOrientationSequence" not in sfg:
        raise ValueError("Plane Orientation (Patient) is missing")

    iop = sfg.PlaneOrientationSequence[0].ImageOrientationPatient
    assert len(iop) == 6

    return dcm_to_sitk_orientation(iop)


def get_image_origin_and_extent(
    dataset: pydicom.Dataset, direction: np.ndarray
) -> Tuple[Tuple[float, ...], float]:
    frames = dataset.PerFrameFunctionalGroupsSequence
    slice_dir = direction[:, 2]
    reference_position = np.asarray(
        [float(x) for x in frames[0].PlanePositionSequence[0].ImagePositionPatient]
    )

    min_distance = 0.0
    origin: Tuple[float, ...] = (0.0, 0.0, 0.0)
    distances: Dict[Tuple, float] = {}
    for frame_idx, frame in enumerate(frames):
        frame_position = tuple(
            float(x) for x in frame.PlanePositionSequence[0].ImagePositionPatient
        )
        if frame_position in distances:
            continue

        frame_distance = np.dot(frame_position - reference_position, slice_dir)  # type: ignore
        distances[frame_position] = frame_distance

        if frame_idx == 0 or frame_distance < min_distance:
            min_distance = frame_distance
            origin = frame_position

    # Sort all distances ascending and compute extent from minimum and
    # maximum distance to reference plane
    distance_values = sorted(distances.values())
    extent = 0.0
    if len(distance_values) > 1:
        extent = abs(distance_values[0] - distance_values[-1])

    return origin, extent


class SegmentationType(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0001)"""

    BINARY = "BINARY"
    FRACTIONAL = "FRACTIONAL"


class SegmentsOverlap(enum.Enum):
    """Possible values for DICOM tag (0x0062, 0x0013)"""

    YES = "YES"
    UNDEFINED = "UNDEFINED"
    NO = "NO"


@dataclass(init=False)
class _ReadResultBase:
    """Base data class for read results.

    Contains common information about a decoded segmentation, e.g. origin,
    voxel spacing and the direction matrix.
    """

    dataset: pydicom.Dataset
    direction: np.ndarray
    origin: Tuple[float, ...]
    segment_infos: Dict[int, pydicom.Dataset]
    size: Tuple[int, ...]
    spacing: Tuple[float, ...]

    @property
    def referenced_series_uid(self) -> str:
        uid: str = self.dataset.ReferencedSeriesSequence[0].SeriesInstanceUID
        return uid

    @property
    def referenced_instance_uids(self) -> List[str]:
        return [
            x.ReferencedSOPInstanceUID
            for x in self.dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        ]


@dataclass(init=False)
class SegmentReadResult(_ReadResultBase):
    """Read result for segment-based decoding of DICOM-SEGs."""

    _segment_data: Dict[int, np.ndarray]

    @property
    def available_segments(self) -> Set[int]:
        return set(self._segment_data.keys())

    def segment_data(self, number: int) -> np.ndarray:
        return self._segment_data[number]

    def segment_image(self, number: int) -> sitk.Image:
        result = sitk.GetImageFromArray(self._segment_data[number])
        result.SetOrigin(self.origin)
        result.SetSpacing(self.spacing)
        result.SetDirection(self.direction.ravel())
        return result


@dataclass(init=False)
class MultiClassReadResult(_ReadResultBase):
    """Read result for multi-class decoding of DICOM-SEGs."""

    data: np.ndarray

    @property
    def image(self) -> sitk.Image:
        result = sitk.GetImageFromArray(self.data)
        result.SetOrigin(self.origin)
        result.SetSpacing(self.spacing)
        result.SetDirection(self.direction.ravel())
        return result
    
class _ReaderBase(abc.ABC):
    """Base class for reader implementations.

    Reading DICOM-SEGs as different output formats still shares a lot of common
    information decoding. This baseclass extracts this common knowledge and
    sets the respective attributes in a `_ReadResultBase` derived result
    instance.
    """

    @abc.abstractmethod
    def read(self, dataset: pydicom.Dataset) -> _ReadResultBase:
        """Read from a DICOM-SEG file.

        Args:
            dataset: A `pydicom.Dataset` with DICOM-SEG content.

        Returns:
            Result object with decoded numpy data and common information about
            the spatial location and extent of the volume.
        """

    def _read_common(self, dataset: pydicom.Dataset, result: _ReadResultBase) -> None:
        """Read common information from a dataset and store it.

        Args:
            dataset: A `pydicom.Dataset` with DICOM-SEG content.
            result: A `_ReadResultBase` derived result object, where the common
                informations are stored.
        """
        if dataset.SOPClassUID != SegmentationStorage or dataset.Modality != "SEG":
            raise ValueError("DICOM dataset is not a DICOM-SEG storage")

        result.dataset = dataset
        result.segment_infos = get_segment_map(dataset)
        result.spacing = get_declared_image_spacing(dataset)
        result.direction = get_image_direction(dataset)
        result.direction.flags.writeable = False
        result.origin, extent = get_image_origin_and_extent(
            dataset, result.direction
        )
        result.size = (
            dataset.Columns,
            dataset.Rows,
            int(np.rint(extent / result.spacing[-1])) + 1,
        )


class SegmentReader(_ReaderBase):
    """Reads binary segments from a DICOM-SEG file.

    All segments in a DICOM-SEG file cover the same spatial extent, but might
    overlap. If a user wants to use each segment individually as a binary
    segmentation, then this reader extracts all segments as individual numpy
    arrays. The read operation creates a `SegmentReadResult` object with common
    information about the spatial location and extent shared by all segments,
    as well as the binary segmentation data for each segment.

    Example:
        ::

            dcm = pydicom.dcmread('segmentation.dcm')
            reader = pydicom_seg.SegmentReader()
            result = reader.read(dcm)
            data = result.segment_data(1)  # numpy array
            image = result.segment_image(1)  # SimpleITK image
    """

    def read(self, dataset: pydicom.Dataset) -> SegmentReadResult:
        result = SegmentReadResult()
        self._read_common(dataset, result)

        # SimpleITK has currently no support for writing slices into memory, allocate a numpy array
        # as intermediate buffer and create an image afterwards
        segmentation_type = SegmentationType[dataset.SegmentationType]
        dtype = np.uint8 if segmentation_type == SegmentationType.BINARY else np.float32

        # pydicom decodes single-frame pixel data without a frame dimension
        frame_pixel_array = dataset.pixel_array
        if dataset.NumberOfFrames == 1 and len(frame_pixel_array.shape) == 2:
            frame_pixel_array = np.expand_dims(frame_pixel_array, axis=0)  # type: ignore

        result._segment_data = {}
        for segment_number in result.segment_infos:
            # Segment buffer should be cleared for each segment since
            # segments may have different number of frames!
            segment_buffer = np.zeros(result.size[::-1], dtype=dtype)

            # Dummy image for computing indices from physical points
            dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
            dummy.SetOrigin(result.origin)
            dummy.SetSpacing(result.spacing)
            dummy.SetDirection(result.direction.ravel())

            # get segment ID sequence for the case it is the same for all frames (e.g. only one segment)
            shared_sis = dataset.SharedFunctionalGroupsSequence[0].get(
                "SegmentIdentificationSequence"
            )

            # Iterate over all frames and check for referenced segment number
            for frame_idx, pffg in enumerate(dataset.PerFrameFunctionalGroupsSequence):
                sis = pffg.get(
                    "SegmentIdentificationSequence", shared_sis
                )  # shared_sis as default value
                if segment_number != sis[0].ReferencedSegmentNumber:
                    continue

                frame_position = [
                    float(x) for x in pffg.PlanePositionSequence[0].ImagePositionPatient
                ]
                frame_index = dummy.TransformPhysicalPointToIndex(frame_position)
                slice_data = frame_pixel_array[frame_idx]

                # If it is fractional data, then convert to range [0, 1]
                if segmentation_type == SegmentationType.FRACTIONAL:
                    slice_data = (
                        slice_data.astype(dtype) / dataset.MaximumFractionalValue
                    )

                segment_buffer[frame_index[2]] = slice_data

            result._segment_data[segment_number] = segment_buffer

        return result
