"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from collections.abc import Mapping
from typing import Union, Any, TypedDict, Optional
from typing_extensions import NotRequired
import zarr.storage
from xarray.core.dataset import Dataset


# ======================================================================================================================
class StoreManagerParameters(TypedDict):
    """User-defined parameters for the creation of a store manager instance. If not set, default values are used."""

    base_fsspec_protocol: NotRequired[str]
    """Base protocol used by 'fsspec'."""

    target_fsspec_protocol: NotRequired[str]
    """Target filesystem protocol for the selected base 'fsspec' protocol."""

    s3_endpoint_url: NotRequired[str]
    """Endpoint of the cloud S3 bucket used for remote data pulling."""

    s3_mast_dataset_path: NotRequired[str]
    """Path for the target MAST dataset within the configured S3 bucket."""

    base_local_zarr_path: NotRequired[Optional[str]]
    """Local root path used for local data pulling in Zarr format."""


# ======================================================================================================================
class ShotInfo(TypedDict):
    """
    Information to pull shot data/metadata from the MAST dataset. The default value for the optional parameter `local`
    is assigned via `MAST_tools.utils.store_utils.MASTStorageManager._parse_shot_info_dict()`.
    """

    shot_id: int
    """ID of a target shot to be pulled from the MAST dataset."""

    local: NotRequired[bool]
    """Boolean flag to define data/metadata location. If True, the target shot is pulled from locally stored data 
    (e.g., in the CSD3 cluster), otherwise it is pulled from the registered remote data repository (e.g., a cloud S3
    bucket)."""


# ======================================================================================================================
# Data types

ShotInfoType = Mapping[str, Any]
StoreManagerParametersType = Mapping[str, Any]

XarrayDatasetType = Dataset
ZarrFSStoreType = zarr.storage.FsspecStore
ZarrLocalStoreType = zarr.storage.LocalStore
ZarrZipStoreType = zarr.storage.ZipStore
ZarrStoreType = Union[ZarrFSStoreType, ZarrLocalStoreType, ZarrZipStoreType]
BaseDataSourceType = Union[ShotInfoType, ZarrStoreType]
ExtendedDataSourceType = Union[BaseDataSourceType, XarrayDatasetType]
